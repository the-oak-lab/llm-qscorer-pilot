"""Stage 2.3 normalisation (`canon`): raw features to canonical observables.

Four steps, each cached in the output folder so a rerun picks up where it stopped.

1. **Embed** every raw feature's "name: definition" (nothing else: no condition, labels,
   question text or mechanism) with `gemini-embedding-2`, the model's clustering prefix in
   front. The vectors are unit-norm.
2. **Cluster** them by Ward's method on the unit vectors. The within-cluster dispersion at k
   clusters follows from the merge heights, and the **elbow** is the knee of that curve over
   k = 1..k_max (the point farthest below the chord, after scaling both axes to [0, 1]). A
   second criterion uses the design's replication: every batch and condition was asked twice,
   so at each cut on a grid the **two-call stability** is the mean Jaccard overlap of the
   canonical feature sets named by the two calls, minus the same overlap between calls on
   different batches of the same condition (shuffled pairings). The elbow is the primary cut;
   the stability-optimal cut is a sensitivity cut, named too when it differs.
3. **Polarity** of each raw feature, since the embedding does not separate "absence of X" from
   "X": a Gemini call classifies raw features in batches as describing the presence or the
   absence of a characteristic.
4. **Name** each cluster of a cut with one Gemini call that sees a sample of its members and
   returns a neutral canonical name and definition of the shared observable, whether the
   members describe one observable, and a note when they do not.

The mapping of a cut gives every raw feature the canonical id `k<cut>-<cluster>` followed by
`+` (presence) or `-` (absence), so the analysis treats the presence and the absence of an
observable as two canonical features of one cluster.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from qscorer.config import CanonParams, DatasetConfig
from qscorer.gemini import Job, parse

KEY = ["batch", "condition", "sample", "feature"]

POLARITY_PROMPT = (
    "Below are descriptions of features of multiple-choice questions or their feedback, each produced by an independent analysis. "
    "For each one, say whether it describes the PRESENCE of a characteristic in the question or feedback, or its ABSENCE (for example "
    "a lack, omission, failure to include, or missing element). Judge the characteristic the description names, not whether it would "
    "help or hinder learning.\n\n{items}\n\n"
    'Respond in JSON with one field, "polarities": a list with one object per feature, each with "index" (the feature\'s number) and '
    '"polarity" ("presence" or "absence").')
POLARITY_SCHEMA = {"type": "object", "required": ["polarities"], "properties": {"polarities": {"type": "array", "items": {
    "type": "object", "required": ["index", "polarity"], "properties": {"index": {"type": "integer"}, "polarity": {"type": "string", "enum": ["presence", "absence"]}}}}}}

NAMING_PROMPT = (
    "Below are {shown} descriptions of a feature of multiple-choice questions or their feedback{sampled}, each produced by an independent "
    "analysis of a few questions. They were grouped together because their wording is similar.\n\n{items}\n\n"
    "Write a canonical name and a one-sentence definition of the observable characteristic these descriptions share. State it "
    "neutrally, as a property of a question or its feedback that is present or absent, without saying whether it helps or hinders "
    "learning, and without reference to any particular science topic. Then say whether the descriptions describe one observable "
    "characteristic (coherent) or several distinct ones; if several, name the distinct characteristics in a short note.\n\n"
    'Respond in JSON with the fields "name", "definition", "coherent" (true or false) and "note" (empty when coherent).')
NAMING_SCHEMA = {"type": "object", "required": ["name", "definition", "coherent", "note"],
                 "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "coherent": {"type": "boolean"}, "note": {"type": "string"}}}


@dataclass
class CanonResult:
    features: pd.DataFrame      # the raw features with their text, polarity and cluster at each named cut
    curves: pd.DataFrame        # k, dispersion, stability (where measured)
    cuts: dict                  # elbow and stability cuts
    clusters: dict[int, pd.DataFrame]   # per named cut: one row per cluster with its canonical name and definition
    mappings: dict[int, pd.DataFrame]   # per named cut: KEY -> canonical id
    summary: dict


def feature_text(features: pd.DataFrame) -> pd.Series:
    return features.name.fillna("").str.strip() + ": " + features.definition.fillna("").str.strip()


def embed(features: pd.DataFrame, params: CanonParams, out: Path, progress=print, client=None) -> np.ndarray:
    """The unit-norm vectors of the raw features, cached in `out/embeddings.npy` with the keys
    they belong to in `out/embedded.csv`."""
    cache, keys = out / "embeddings.npy", out / "embedded.csv"
    if cache.exists() and keys.exists():
        X = np.load(cache)
        k = pd.read_csv(keys)
        if len(X) == len(features) and (k[KEY].to_numpy() == features[KEY].to_numpy()).all():
            progress(f"embeddings: {len(X)} cached")
            return X
    texts = feature_text(features).tolist()
    if client is None:
        from google import genai
        from qscorer.gemini import api_key, embed_many
        client = genai.Client(api_key=api_key())
        vectors = asyncio.run(embed_many(client, params.embedding_model, texts, params.embed_batch, params.embed_concurrency, params.prefix, progress))
    else:
        vectors = client.embed(texts)
    X = np.asarray(vectors, dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    np.save(cache, X)
    features[KEY].to_csv(keys, index=False)
    return X


def dispersion_curve(Z: np.ndarray, k_max: int) -> pd.DataFrame:
    """Within-cluster sum of squares at k = 1..k_max clusters under Ward's linkage: each merge
    adds half the squared merge height."""
    n = len(Z) + 1
    gains = 0.5 * Z[:, 2] ** 2                    # merge i (i = 0..n-2) takes the count from n - i to n - i - 1
    cum = np.concatenate([[0.0], np.cumsum(gains)])  # cum[m]: dispersion inside the clusters after the first m merges
    k = np.arange(1, min(k_max, n) + 1)
    return pd.DataFrame({"k": k, "dispersion": cum[n - k]})


def knee(k: np.ndarray, y: np.ndarray) -> int:
    """The knee of a decreasing convex curve: the point farthest below the chord from its first
    to its last point, both axes scaled to [0, 1]."""
    x = (k - k[0]) / max(k[-1] - k[0], 1)
    yy = (y - y[-1]) / max(y[0] - y[-1], 1e-12)
    chord = 1 - x
    return int(k[np.argmax(chord - yy)])


def cut(Z: np.ndarray, k: int) -> np.ndarray:
    """Cluster labels 1..k."""
    return fcluster(Z, k, criterion="maxclust")


def stability(features: pd.DataFrame, labels: np.ndarray, shuffles: int, rng: np.random.Generator) -> tuple[float, float]:
    """Mean Jaccard overlap of the canonical sets named by the two calls on the same batch and
    condition, and the same between calls on different batches of the same condition."""
    f = features[["batch", "condition", "sample"]].assign(label=labels)
    sets = f.groupby(["batch", "condition", "sample"]).label.agg(lambda s: frozenset(s)).unstack("sample")
    sets = sets.dropna()
    if sets.shape[1] < 2 or not len(sets):
        return float("nan"), float("nan")
    a, b = sets.iloc[:, 0].to_numpy(), sets.iloc[:, 1].to_numpy()
    jac = lambda s, t: len(s & t) / len(s | t) if (s | t) else 1.0
    observed = float(np.mean([jac(s, t) for s, t in zip(a, b)]))
    conditions = sets.index.get_level_values("condition").to_numpy()
    base = []
    for _ in range(shuffles):
        perm = np.arange(len(b))
        for c in np.unique(conditions):
            idx = np.flatnonzero(conditions == c)
            if len(idx) > 1:
                perm[idx] = idx[rng.permutation(len(idx))]
        base.append(np.mean([jac(s, t) for s, t in zip(a, b[perm])]))
    return observed, float(np.mean(base))


def polarity_jobs(features: pd.DataFrame, params: CanonParams, config: DatasetConfig) -> list[Job]:
    texts = feature_text(features).tolist()
    jobs = []
    for start in range(0, len(texts), params.polarity_batch):
        items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts[start:start + params.polarity_batch]))
        jobs.append(Job(f"polarity|{start}", POLARITY_PROMPT.format(items=items), gemini_config(params, config, POLARITY_SCHEMA, seed=0)))
    return jobs


def gemini_config(params: CanonParams, config: DatasetConfig, schema: dict, seed: int) -> dict:
    return {"system_instruction": config.prompt.system, "response_mime_type": "application/json", "response_json_schema": schema,
            "temperature": params.temperature, "seed": seed, "max_output_tokens": params.max_output_tokens,
            "thinking_config": {"thinking_level": params.thinking.upper(), "include_thoughts": True}}


def parse_polarities(text: str | None, n: int) -> dict[int, str] | None:
    """index (1-based) -> polarity, when every index 1..n is answered; else None."""
    if not text:
        return None
    try:
        items = json.loads(text).get("polarities")
    except (ValueError, AttributeError):
        return None
    if not isinstance(items, list):
        return None
    out = {}
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("index"), int) and it.get("polarity") in ("presence", "absence"):
            out[it["index"]] = it["polarity"]
    return out if all(i in out for i in range(1, n + 1)) else None


def accept_polarity(raw: dict, n_by_id: dict[str, int]) -> bool:
    r = parse(raw)
    return r.ok and parse_polarities(r.text, n_by_id.get(raw["id"], 0)) is not None


def naming_jobs(features: pd.DataFrame, labels: np.ndarray, k: int, params: CanonParams, config: DatasetConfig, rng: np.random.Generator) -> list[Job]:
    texts = feature_text(features).to_numpy()
    jobs = []
    for c in range(1, k + 1):
        members = np.flatnonzero(labels == c)
        shown = members if len(members) <= params.members_shown else np.sort(rng.choice(members, params.members_shown, replace=False))
        items = "\n".join(f"- {texts[i]}" for i in shown)
        sampled = f" (a sample of {len(members)} such descriptions)" if len(members) > len(shown) else ""
        jobs.append(Job(f"name|{k}|{c}", NAMING_PROMPT.format(shown=len(shown), sampled=sampled, items=items), gemini_config(params, config, NAMING_SCHEMA, seed=0)))
    return jobs


def parse_name(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        d = json.loads(text)
        if all(k in d for k in ("name", "definition", "coherent")) and d["name"] and d["definition"]:
            return {"name": str(d["name"]), "definition": str(d["definition"]), "coherent": bool(d["coherent"]), "note": str(d.get("note", ""))}
    except (ValueError, AttributeError, TypeError):
        pass
    return None


def accept_name(raw: dict) -> bool:
    r = parse(raw)
    return r.ok and parse_name(r.text) is not None


def run_canon(config: DatasetConfig, params: CanonParams, features: pd.DataFrame, out: Path, engine=None, progress=print) -> CanonResult:
    """The four steps. `engine`, for tests, provides `embed(texts)` and `many(jobs, log, passes, accept, progress)`."""
    out.mkdir(parents=True, exist_ok=True)
    features = features.reset_index(drop=True)
    rng = np.random.default_rng(params.seed)
    X = embed(features, params, out, progress, client=engine)
    Z = linkage(X, method="ward")
    np.save(out / "linkage.npy", Z)
    curves = dispersion_curve(Z, params.k_max)
    elbow = knee(curves.k.to_numpy(), curves.dispersion.to_numpy())
    stab = []
    for k in params.k_grid:
        if k <= len(features):
            observed, baseline = stability(features, cut(Z, k), params.stability_shuffles, rng)
            stab.append({"k": k, "stability_observed": observed, "stability_baseline": baseline, "stability": observed - baseline})
    stab = pd.DataFrame(stab)
    curves = curves.merge(stab, on="k", how="outer").sort_values("k")
    best = int(stab.loc[stab.stability.idxmax(), "k"]) if len(stab) and stab.stability.notna().any() else elbow
    cuts = {"elbow": elbow, "stability": best, "k_max": params.k_max, "named": sorted({elbow, best})}
    progress(f"clusters: elbow at k = {elbow}, stability-optimal cut at k = {best}")

    log = out / "responses.jsonl"
    if engine is None:
        from qscorer.gemini import Gemini
        gem = Gemini(params.model, concurrency=params.concurrency, delay=params.delay, timeout=params.timeout)
        many = lambda jobs, accept: asyncio.run(gem.many(jobs, log, params.passes, accept, progress))
    else:
        many = lambda jobs, accept: engine.many(jobs, log, params.passes, accept, progress)

    pjobs = polarity_jobs(features, params, config)
    n_by_id = {j.id: min(params.polarity_batch, len(features) - int(j.id.split("|")[1])) for j in pjobs}
    records = many(pjobs, lambda raw: accept_polarity(raw, n_by_id))
    polarity = np.full(len(features), "", dtype=object)
    for j in pjobs:
        start = int(j.id.split("|")[1])
        answers = parse_polarities(parse(records[j.id]).text, n_by_id[j.id]) if j.id in records else None
        for i in range(n_by_id[j.id]):
            polarity[start + i] = answers[i + 1] if answers else "unknown"
    features = features.assign(text=feature_text(features), polarity=polarity)

    clusters, mappings = {}, {}
    for k in cuts["named"]:
        labels = cut(Z, k)
        features[f"cluster_k{k}"] = labels
        jobs = naming_jobs(features, labels, k, params, config, rng)
        records = many(jobs, accept_name)
        rows = []
        for j in jobs:
            c = int(j.id.split("|")[2])
            members = features[labels == c]
            named = parse_name(parse(records[j.id]).text) if j.id in records else None
            rows.append({"cut": k, "cluster": c, "size": int(len(members)), "absence_share": float((members.polarity == "absence").mean()),
                         "conditions": " ".join(f"{cond}:{n}" for cond, n in members.condition.value_counts().sort_index().items()),
                         "name": named["name"] if named else "", "definition": named["definition"] if named else "",
                         "coherent": named["coherent"] if named else None, "note": named["note"] if named else "(naming call failed)",
                         "example_members": " | ".join(members.text.head(3))})
        clusters[k] = pd.DataFrame(rows)
        sign = features.polarity.map({"presence": "+", "absence": "-"}).fillna("")
        mappings[k] = features[KEY].assign(canonical=np.where(sign.ne(""), "k" + str(k) + "-" + pd.Series(labels).astype(str).str.zfill(3) + sign, ""))
    summary = {"dataset": config.name, "params": params.model_dump(mode="json"), "raw_features": int(len(features)), "dimensions": int(X.shape[1]),
               "cuts": cuts, "polarity": features.polarity.value_counts().to_dict(),
               "clusters": {str(k): {"n": int(len(t)), "coherent": int(t.coherent.fillna(False).sum()), "named": int(t.name.ne("").sum()),
                                     "largest": int(t["size"].max()), "singletons": int((t["size"] == 1).sum())} for k, t in clusters.items()},
               "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    return CanonResult(features, curves, cuts, clusters, mappings, summary)
