"""Stage 2.3 normalisation (`canon`): raw features to a checklist of observables.

Five steps, each cached in the output folder so a rerun picks up where it stopped.

1. **Embed** every raw feature's "name: definition" (nothing else: no condition, labels,
   question text or mechanism) with the embedding model, the clustering prefix in front. The
   vectors are unit-norm.
2. **Cluster** them by Ward's method on the unit vectors and cut the tree at a fixed, coarse
   `k` (default 50). The number is a budget, not a discovery: the dispersion curve of these
   embeddings has no knee (an elbow found by the chord method moves with the largest k
   considered), and Ward splits popular observables by size, so the cut is followed by naming
   and a consolidation that merges what it split. The design's replication gives a diagnostic
   of the cut: every batch and condition was asked twice, so the **two-call stability** is the
   mean Jaccard overlap of the cluster sets named by the two calls, minus the same overlap
   between calls on different batches of the same condition (shuffled pairings).
3. **Name** each cluster with one call that sees a sample of its members and returns a name in
   presence form, a neutral definition, a scoring criterion, the part of the item concerned,
   whether the members describe one observable, and the distinct ones when they do not.
4. **Split** each incoherent cluster once into its two Ward children (when both have at least
   `min_split` members) and name the children. The clusters not split and the children of the
   split ones are the **leaves**.
5. **Consolidate** the leaves in one call that sees every leaf's name, definition and criterion
   and merges the leaves that describe one observable (synonyms, and the presence and the
   absence of one characteristic) into checklist **items**. Every leaf belongs to one item;
   leaves the call leaves out become items of their own.

The checklist (items with name, definition, criterion, part and source leaves) is what Stage 3
scores questions against. The mapping gives every raw feature its item id, so the Stage 2.4
contrasts can run on the same partition.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, to_tree

from qscorer.config import CanonParams, DatasetConfig
from qscorer.llm import Job, make_backend, parse

KEY = ["batch", "condition", "sample", "feature"]
PARTS = ("stem", "options", "feedback", "cross-item")

NAMING_PROMPT = (
    "Below are {shown} descriptions of a feature of multiple-choice questions or their feedback{sampled}, each produced by an independent "
    "analysis of a few questions. They were grouped together because their wording is similar.\n\n{items}\n\n"
    "Identify the observable characteristic whose presence or absence these descriptions report. Give it a canonical name in presence form "
    "(name the characteristic itself, not its absence), a one-sentence definition stated neutrally as a property of a question or its feedback, "
    "without saying whether it helps or hinders learning and without reference to any particular science topic, and a one-sentence scoring "
    "criterion (count the characteristic as present in a question when ...). Say which part of the item it concerns. Then say whether the "
    "descriptions report one characteristic (coherent) or several distinct ones; if several, list the distinct characteristics.\n\n"
    'Respond in JSON with the fields "name", "definition", "criterion", "part" ("stem", "options", "feedback" or "cross-item"), '
    '"coherent" (true or false) and "sub_characteristics" (a list of short names, empty when coherent).')
NAMING_SCHEMA = {"type": "object", "required": ["name", "definition", "criterion", "part", "coherent", "sub_characteristics"],
                 "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "criterion": {"type": "string"},
                                "part": {"type": "string", "enum": list(PARTS)}, "coherent": {"type": "boolean"},
                                "sub_characteristics": {"type": "array", "items": {"type": "string"}}}}

CONSOLIDATE_PROMPT = (
    "Below is a list of {n} candidate features of multiple-choice questions or their feedback, each with an id, a name, a definition and a "
    "scoring criterion. They were derived by clustering descriptions produced by independent analyses, so some entries describe the same "
    "observable characteristic in different words, and some describe the presence and others the absence of one characteristic.\n\n{items}\n\n"
    "Consolidate them into a checklist of distinct observable characteristics. Merge entries only when one rubric scoring the presence of a "
    "single characteristic would score all of them; the presence and the absence of one characteristic are the same entry. Do not merge "
    "characteristics that are related but distinct, where a question could plausibly exhibit one without the other. Every entry must be "
    "assigned to exactly one checklist item. For each item give a name in presence form, a one-sentence definition stated neutrally without "
    "reference to any science topic, a one-sentence scoring criterion (count as present when ...), the part of the item it concerns, and the "
    "ids of the entries it merges.\n\n"
    'Respond in JSON with one field, "items": a list of objects with "name", "definition", "criterion", "part" ("stem", "options", '
    '"feedback" or "cross-item") and "sources" (a list of entry ids).')
CONSOLIDATE_SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["name", "definition", "criterion", "part", "sources"],
    "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "criterion": {"type": "string"},
                   "part": {"type": "string", "enum": list(PARTS)}, "sources": {"type": "array", "items": {"type": "string"}}}}}}}


@dataclass
class CanonResult:
    features: pd.DataFrame      # the raw features with their text, cluster, leaf and item
    leaves: pd.DataFrame        # one row per leaf (cluster or child) with its name, definition, criterion and coherence
    checklist: list[dict]       # the items: id, name, definition, criterion, part, sources (leaf ids), size
    mapping: pd.DataFrame       # KEY -> canonical (the item id)
    summary: dict


def feature_text(features: pd.DataFrame) -> pd.Series:
    return features.name.fillna("").str.strip() + ": " + features.definition.fillna("").str.strip()


def embed(features: pd.DataFrame, params: CanonParams, out: Path, engine, progress=print) -> np.ndarray:
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
    vectors = engine.embed(texts, params.embedding_model, params.embed_batch, params.embed_concurrency, params.prefix, progress)
    X = np.asarray(vectors, dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    np.save(cache, X)
    features[KEY].to_csv(keys, index=False)
    return X


def cut(Z: np.ndarray, k: int) -> np.ndarray:
    """Cluster labels 1..k."""
    return fcluster(Z, k, criterion="maxclust")


def children(Z: np.ndarray, members: np.ndarray) -> list[np.ndarray]:
    """The two Ward children of the cluster whose members these are (sorted indices), or the
    cluster alone when it is a single feature."""
    _, nodes = to_tree(Z, rd=True)
    target = set(int(i) for i in members)
    for node in nodes:
        if node.count == len(target) and not node.is_leaf() and set(node.pre_order()) == target:
            return [np.array(sorted(node.left.pre_order())), np.array(sorted(node.right.pre_order()))]
    return [np.asarray(members)]


def stability(features: pd.DataFrame, labels: np.ndarray, shuffles: int, rng: np.random.Generator) -> tuple[float, float]:
    """Mean Jaccard overlap of the cluster sets named by the two calls on the same batch and
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


def job(id: str, prompt: str, schema: dict, params: CanonParams, config: DatasetConfig, thinking: str | None = None, max_output_tokens: int | None = None) -> Job:
    return Job(id, prompt, system=config.prompt.system, schema=schema, temperature=params.temperature, seed=0,
               max_output_tokens=max_output_tokens or params.max_output_tokens, thinking=thinking or params.thinking)


def naming_jobs(groups: dict[str, np.ndarray], texts: np.ndarray, params: CanonParams, config: DatasetConfig, rng: np.random.Generator) -> list[Job]:
    """One naming call per group (leaf id -> member indices)."""
    jobs = []
    for gid, members in groups.items():
        shown = members if len(members) <= params.members_shown else np.sort(rng.choice(members, params.members_shown, replace=False))
        items = "\n".join(f"- {texts[i]}" for i in shown)
        sampled = f" (a sample of {len(members)} such descriptions)" if len(members) > len(shown) else ""
        jobs.append(job(f"name|{gid}", NAMING_PROMPT.format(shown=len(shown), sampled=sampled, items=items), NAMING_SCHEMA, params, config))
    return jobs


def parse_name(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        d = json.loads(text)
        if all(d.get(k) for k in ("name", "definition", "criterion")) and d.get("part") in PARTS and isinstance(d.get("coherent"), bool):
            subs = d.get("sub_characteristics") or []
            return {"name": str(d["name"]), "definition": str(d["definition"]), "criterion": str(d["criterion"]), "part": str(d["part"]),
                    "coherent": bool(d["coherent"]), "sub_characteristics": [str(s) for s in subs if str(s).strip()]}
    except (ValueError, AttributeError, TypeError):
        pass
    return None


def accept_name(raw: dict) -> bool:
    r = parse(raw)
    return r.ok and parse_name(r.text) is not None


def consolidate_job(leaves: pd.DataFrame, params: CanonParams, config: DatasetConfig) -> Job:
    items = "\n".join(f"{r.leaf}. {r.name}: {r.definition} Count as present when {r.criterion.rstrip('.')}." if not r.criterion.lower().startswith("count")
                      else f"{r.leaf}. {r.name}: {r.definition} {r.criterion}" for r in leaves.itertuples())
    return job("consolidate|0", CONSOLIDATE_PROMPT.format(n=len(leaves), items=items), CONSOLIDATE_SCHEMA, params, config,
               thinking=params.consolidate_thinking, max_output_tokens=max(params.max_output_tokens, 65536))


def parse_checklist(text: str | None, leaf_ids: list[str], min_coverage: float) -> list[dict] | None:
    """The items with their sources restricted to known leaves, each leaf kept with its first
    item; None unless at least `min_coverage` of the leaves are assigned."""
    if not text:
        return None
    try:
        raw = json.loads(text).get("items")
    except (ValueError, AttributeError):
        return None
    if not isinstance(raw, list):
        return None
    known, seen, items = set(leaf_ids), set(), []
    for it in raw:
        if not isinstance(it, dict) or not all(it.get(k) for k in ("name", "definition", "criterion")) or it.get("part") not in PARTS:
            continue
        sources = [str(s).strip() for s in (it.get("sources") or [])]
        sources = [s for s in sources if s in known and s not in seen]
        if not sources:
            continue
        seen.update(sources)
        items.append({"name": str(it["name"]), "definition": str(it["definition"]), "criterion": str(it["criterion"]), "part": str(it["part"]), "sources": sources})
    return items if len(seen) >= min_coverage * len(leaf_ids) else None


def run_canon(config: DatasetConfig, params: CanonParams, features: pd.DataFrame, out: Path, engine=None, progress=print) -> CanonResult:
    """The five steps. `engine` is the backend (default: the one `params` names); a test double provides
    `embed(texts, ...)`, `many(jobs, log, passes, accept, progress)` and `describe()`."""
    out.mkdir(parents=True, exist_ok=True)
    engine = engine or make_backend(params)
    features = features.reset_index(drop=True)
    rng = np.random.default_rng(params.seed)
    texts = feature_text(features).to_numpy()
    X = embed(features, params, out, engine, progress)
    Z = linkage(X, method="ward")
    np.save(out / "linkage.npy", Z)
    k = min(params.k, len(features))
    labels = cut(Z, k)
    observed, baseline = stability(features, labels, params.stability_shuffles, rng)
    progress(f"clusters: {k} at the cut, two-call stability {observed - baseline:.3f} ({observed:.2f} observed, {baseline:.2f} shuffled)")
    log = out / "responses.jsonl"
    many = lambda jobs, accept: engine.many(jobs, log, params.passes, accept, progress)

    # name the clusters
    groups = {f"c{c}": np.flatnonzero(labels == c) for c in range(1, k + 1)}
    records = many(naming_jobs(groups, texts, params, config, rng), accept_name)
    named = {gid: parse_name(parse(records[f"name|{gid}"]).text) if f"name|{gid}" in records else None for gid in groups}
    # split the incoherent ones once, name the children
    todo = {}
    for gid, nd in named.items():
        if nd is not None and not nd["coherent"]:
            kids = children(Z, groups[gid])
            if len(kids) == 2 and all(len(x) >= params.min_split for x in kids):
                for t, x in enumerate(kids):
                    todo[f"{gid}.{t + 1}"] = x
    progress(f"naming: {sum(nd is not None for nd in named.values())} of {k} clusters named, {sum(nd is not None and not nd['coherent'] for nd in named.values())} incoherent, "
             f"{len(todo) // 2} split into children")
    records = many(naming_jobs(todo, texts, params, config, rng), accept_name)
    for gid, members in todo.items():
        groups[gid] = members
        named[gid] = parse_name(parse(records[f"name|{gid}"]).text) if f"name|{gid}" in records else None
    split = {gid[:gid.rindex(".")] for gid in todo}
    leaf_ids = [gid for gid in groups if gid not in split]
    leaf = np.full(len(features), "", dtype=object)
    for gid in leaf_ids:
        leaf[groups[gid]] = gid
    rows = []
    for gid in leaf_ids:
        members = features.iloc[groups[gid]]
        nd = named[gid] or {"name": "", "definition": "", "criterion": "", "part": "", "coherent": None, "sub_characteristics": []}
        rows.append({"leaf": gid, "cluster": gid.split(".")[0], "size": int(len(members)), **{key: nd[key] for key in ("name", "definition", "criterion", "part", "coherent")},
                     "sub_characteristics": " | ".join(nd["sub_characteristics"]),
                     "conditions": " ".join(f"{cond}:{n}" for cond, n in members.condition.value_counts().sort_index().items()),
                     "example_members": " | ".join(members.text.head(3)) if "text" in members else " | ".join(texts[groups[gid][:3]])})
    leaves = pd.DataFrame(rows)
    unnamed = leaves[leaves.name.eq("")]
    if len(unnamed):
        progress(f"warning: {len(unnamed)} leaves without a name are left out of the consolidation and become items of their own")

    # consolidate
    namedleaves = leaves[leaves.name.ne("")]
    cj = consolidate_job(namedleaves, params, config)
    ids = namedleaves.leaf.tolist()
    records = many([cj], lambda raw: parse(raw).ok and parse_checklist(parse(raw).text, ids, params.min_coverage) is not None)
    items = parse_checklist(parse(records[cj.id]).text, ids, params.min_coverage) if cj.id in records else None
    if items is None:
        progress("warning: the consolidation call gave no acceptable answer; every leaf becomes an item")
        items = []
    assigned = {s for it in items for s in it["sources"]}
    for r in leaves.itertuples():
        if r.leaf not in assigned:
            items.append({"name": r.name or f"leaf {r.leaf}", "definition": r.definition, "criterion": r.criterion, "part": r.part, "sources": [r.leaf]})
    by_leaf = {}
    checklist = []
    for t, it in enumerate(items):
        iid = f"f{t + 1:02d}"
        size = int(leaves.set_index("leaf")["size"][it["sources"]].sum())
        checklist.append({"id": iid, **it, "size": size})
        for s in it["sources"]:
            by_leaf[s] = iid
    item = pd.Series(leaf).map(by_leaf).fillna("").to_numpy()
    features = features.assign(text=texts, cluster=labels, leaf=leaf, item=item)
    leaves["item"] = leaves.leaf.map(by_leaf)
    mapping = features[KEY].assign(canonical=item)
    progress(f"checklist: {len(checklist)} items from {len(leaves)} leaves; {sum(len(it['sources']) > 1 for it in checklist)} items merge more than one leaf")
    summary = {"dataset": config.name, "params": params.model_dump(mode="json"), "backend": engine.describe(),
               "raw_features": int(len(features)), "dimensions": int(X.shape[1]), "k": k,
               "stability": {"observed": observed, "shuffled": baseline, "difference": observed - baseline},
               "clusters": {"named": int(sum(named[g] is not None for g in groups if "." not in g)), "incoherent": int(sum(named[g] is not None and not named[g]["coherent"] for g in groups if "." not in g)),
                            "split": len(split)},
               "leaves": {"n": int(len(leaves)), "coherent": int(leaves.coherent.fillna(False).astype(bool).sum()), "unnamed": int(len(unnamed))},
               "checklist": {"n": len(checklist), "merged_items": int(sum(len(it["sources"]) > 1 for it in checklist)), "largest": max(it["size"] for it in checklist) if checklist else 0,
                             "by_part": pd.Series([it["part"] for it in checklist]).value_counts().to_dict()},
               "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    return CanonResult(features, leaves, checklist, mapping, summary)
