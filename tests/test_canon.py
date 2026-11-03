"""The normalisation without the network: a fake embedder with planted clusters, the dispersion curve and knee, the stability
measure, polarity and naming parsing, and the run end to end with a fake engine."""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from qscorer.canon import cut, dispersion_curve, knee, parse_name, parse_polarities, run_canon, stability
from qscorer.config import CanonParams, DatasetConfig, PhaseFilter
from tests.test_baseline import raw

CONFIG = DatasetConfig(name="fake", transactions="x.txt", questions="x.jsonl", phases=PhaseFilter(practice="^learning$"))
GROUPS = ["distractor refutation in feedback", "lexical overlap between stem and key", "worked computation in feedback"]


def fake_features(n_batches=6, samples=2, seed=0):
    """Each batch's calls name one feature from each of two of the three families (batch b skips family b mod 3), so the two calls on a
    batch agree with each other more than with other batches' calls; family 0 alternates presence and absence."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batches):
        for s in range(samples):
            for i, g in enumerate(GROUPS):
                if i == b % 3:
                    continue
                absent = i == 0 and (b + s) % 2 == 0
                name = ("Absence of " if absent else "") + g
                rows.append({"batch": f"b{b}", "condition": "E5", "sample": s, "feature": i + 1, "name": name, "definition": f"Variant {rng.integers(100)} of {g}."})
    return pd.DataFrame(rows)


class Engine:
    """Embeds by family (a family centre plus noise); answers polarity from the name and names clusters from their first member."""
    def __init__(self):
        rng = np.random.default_rng(1)
        self.centres = rng.normal(size=(len(GROUPS), 16))
        self.texts = None

    def describe(self):
        return {"backend": "fake"}

    def embed(self, texts, *args):
        self.texts = texts
        rng = np.random.default_rng(2)
        out = []
        for t in texts:
            g = next(i for i, name in enumerate(GROUPS) if name in t)
            out.append(self.centres[g] + 0.05 * rng.normal(size=16))
        return out

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            kind = j.id.split("|")[0]
            if kind == "polarity":
                lines = [l for l in j.prompt.split("\n") if l[:1].isdigit()]
                answer = {"polarities": [{"index": int(l.split(".")[0]), "polarity": "absence" if "Absence" in l else "presence"} for l in lines]}
            else:
                first = j.prompt.split("\n- ")[1].split(":")[0]
                answer = {"name": first.replace("Absence of ", ""), "definition": "The feedback or question has this property.", "coherent": True, "note": ""}
            r = raw(j.id, text=json.dumps(answer))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_curve_knee_and_stability():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(loc=c, scale=0.02, size=(30, 4)) for c in (np.array([1, 0, 0, 0.]), np.array([0, 1, 0, 0.]), np.array([0, 0, 1, 0.]))])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = linkage(X, method="ward")
    curve = dispersion_curve(Z, 20)
    assert curve.k.tolist() == list(range(1, 21)) and curve.dispersion.is_monotonic_decreasing and curve.dispersion.iloc[0] > 10 * curve.dispersion.iloc[2]
    assert knee(curve.k.to_numpy(), curve.dispersion.to_numpy()) == 3
    labels = cut(Z, 3)
    assert len(set(labels[:30])) == 1 and len(set(labels)) == 3
    f = pd.DataFrame({"batch": np.repeat([f"b{i}" for i in range(15)], 6), "condition": "E5", "sample": np.tile([0, 0, 0, 1, 1, 1], 15)})
    same = np.tile([1, 2, 3], 30)
    observed, baseline = stability(f, same, 5, rng)
    assert observed == 1.0 and baseline == 1.0                      # every call names all three: trivial agreement at a coarse cut
    fine = np.arange(1, 91)
    observed, baseline = stability(f, fine, 5, rng)
    assert observed == 0.0 and baseline == 0.0                      # every feature its own cluster: no agreement at all
    half = np.where(np.tile([0, 0, 0, 1, 1, 1], 15) == 0, np.tile([1, 2, 3], 30), np.tile([1, 2, 9], 30))
    observed, baseline = stability(f, half, 5, rng)
    assert abs(observed - 0.5) < 1e-9                               # two of three shared, four distinct: Jaccard 2/4


def test_parsers():
    assert parse_polarities(json.dumps({"polarities": [{"index": 1, "polarity": "absence"}, {"index": 2, "polarity": "presence"}]}), 2) == {1: "absence", 2: "presence"}
    assert parse_polarities(json.dumps({"polarities": [{"index": 1, "polarity": "absence"}]}), 2) is None and parse_polarities("x", 1) is None
    assert parse_name(json.dumps({"name": "n", "definition": "d", "coherent": False, "note": "two things"})) == {"name": "n", "definition": "d", "coherent": False, "note": "two things"}
    assert parse_name(json.dumps({"name": "", "definition": "d", "coherent": True})) is None


def test_run_canon_with_a_fake_engine(tmp_path):
    features = fake_features()
    params = CanonParams(k_max=10, k_grid=[2, 3, 6, 10], stability_shuffles=3, polarity_batch=7, members_shown=5)
    engine = Engine()
    result = run_canon(CONFIG, params, features, tmp_path, engine=engine, progress=lambda s: None)
    assert engine.texts[0].startswith("lexical overlap between stem and key: Variant") and len(engine.texts) == 24
    assert result.cuts["elbow"] == 3 and result.cuts["stability"] == 3 and result.cuts["named"] == [3]
    assert result.features.polarity.tolist().count("absence") == 4 and set(result.features.polarity) == {"presence", "absence"}
    c = result.clusters[3]
    assert len(c) == 3 and c.coherent.all() and set(c.name) == set(GROUPS) and c["size"].tolist() == [8, 8, 8]
    assert abs(c.set_index("name").loc[GROUPS[0], "absence_share"] - 0.5) < 1e-9
    m = result.mappings[3]
    assert m.canonical.str.match(r"k3-\d{3}[+-]").all() and m.canonical.nunique() == 4      # three clusters, one of them split by polarity
    assert (tmp_path / "embeddings.npy").exists() and (tmp_path / "linkage.npy").exists()
    again = run_canon(CONFIG, params, features, tmp_path, engine=engine, progress=lambda s: None)   # the embeddings come from the cache
    assert again.cuts == result.cuts
