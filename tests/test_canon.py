"""The normalisation without the network: a fake embedder with planted families, the stability measure, the parsers, the Ward children,
and the run end to end with a fake engine that names clusters from their first member and consolidates by name."""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from qscorer.canon import children, cut, parse_checklist, parse_name, run_canon, stability
from qscorer.config import CanonParams, DatasetConfig, PhaseFilter
from tests.test_baseline import raw

CONFIG = DatasetConfig(name="fake", transactions="x.txt", questions="x.jsonl", phases=PhaseFilter(practice="^learning$"))
GROUPS = ["distractor refutation in feedback", "lexical overlap between stem and key", "worked computation in feedback"]


def fake_features(n_batches=6, samples=2, seed=0):
    """Each batch's calls name one feature from each of two of the three families (batch b skips family b mod 3); family 0 alternates
    presence and absence, so a cut into four clusters separates its two polarities and the consolidation must merge them back."""
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
    """Embeds by family and polarity (a centre plus noise); names a group from its first member; consolidates leaves by family."""
    def __init__(self, incoherent: set[str] = frozenset()):
        rng = np.random.default_rng(1)
        self.centres = rng.normal(size=(len(GROUPS) + 1, 16))
        self.texts = None
        self.incoherent = incoherent      # names of groups answered as incoherent
        self.consolidations = 0

    def describe(self):
        return {"backend": "fake"}

    def embed(self, texts, *args):
        self.texts = texts
        rng = np.random.default_rng(2)
        out = []
        for t in texts:
            g = next(i for i, name in enumerate(GROUPS) if name in t)
            c = self.centres[g] + (0.6 * self.centres[3] if t.startswith("Absence") else 0)
            out.append(c + 0.05 * rng.normal(size=16))
        return out

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            kind = j.id.split("|")[0]
            if kind == "name":
                first = j.prompt.split("\n- ")[1].split(":")[0]
                name = first.replace("Absence of ", "")
                answer = {"name": name, "definition": f"The item has {name}.", "criterion": f"it has {name}", "part": "feedback",
                          "coherent": name not in self.incoherent, "sub_characteristics": [] if name not in self.incoherent else ["a", "b"]}
            else:
                self.consolidations += 1
                lines = [l for l in j.prompt.split("\n") if l.startswith("c")]
                by_name = {}
                for l in lines:
                    lid, rest = l.split(". ", 1)
                    by_name.setdefault(rest.split(":")[0], []).append(lid)
                answer = {"items": [{"name": n, "definition": f"The item has {n}.", "criterion": f"it has {n}", "part": "feedback", "sources": ids}
                                    for n, ids in by_name.items()]}
            r = raw(j.id, text=json.dumps(answer))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_stability_and_children():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(loc=c, scale=0.02, size=(30, 4)) for c in (np.array([1, 0, 0, 0.]), np.array([0, 1, 0, 0.]), np.array([0, 0, 1, 0.]))])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = linkage(X, method="ward")
    labels = cut(Z, 3)
    assert len(set(labels[:30])) == 1 and len(set(labels)) == 3
    kids = children(Z, np.flatnonzero(labels == labels[0]))
    assert len(kids) == 2 and sorted(np.concatenate(kids).tolist()) == list(range(30))
    assert len(children(Z, np.array([5]))) == 1
    f = pd.DataFrame({"batch": np.repeat([f"b{i}" for i in range(15)], 6), "condition": "E5", "sample": np.tile([0, 0, 0, 1, 1, 1], 15)})
    observed, baseline = stability(f, np.tile([1, 2, 3], 30), 5, rng)
    assert observed == 1.0 and baseline == 1.0                      # every call names all three: trivial agreement at a coarse cut
    observed, baseline = stability(f, np.arange(1, 91), 5, rng)
    assert observed == 0.0 and baseline == 0.0                      # every feature its own cluster: no agreement at all
    half = np.where(np.tile([0, 0, 0, 1, 1, 1], 15) == 0, np.tile([1, 2, 3], 30), np.tile([1, 2, 9], 30))
    observed, baseline = stability(f, half, 5, rng)
    assert abs(observed - 0.5) < 1e-9                               # two of three shared, four distinct: Jaccard 2/4


def test_parsers():
    good = {"name": "n", "definition": "d", "criterion": "c", "part": "stem", "coherent": False, "sub_characteristics": ["x", ""]}
    assert parse_name(json.dumps(good)) == {"name": "n", "definition": "d", "criterion": "c", "part": "stem", "coherent": False, "sub_characteristics": ["x"]}
    assert parse_name(json.dumps(good | {"part": "elsewhere"})) is None and parse_name(json.dumps(good | {"name": ""})) is None and parse_name("x") is None
    items = {"items": [{"name": "a", "definition": "d", "criterion": "c", "part": "stem", "sources": ["c1", "c2", "zz"]},
                       {"name": "b", "definition": "d", "criterion": "c", "part": "options", "sources": ["c2", "c3"]}]}
    got = parse_checklist(json.dumps(items), ["c1", "c2", "c3", "c4"], 0.75)
    assert [it["sources"] for it in got] == [["c1", "c2"], ["c3"]]      # unknown leaf dropped, a leaf kept with its first item
    assert parse_checklist(json.dumps(items), ["c1", "c2", "c3", "c4"], 0.9) is None   # c4 unassigned: below the coverage required


def test_run_canon_with_a_fake_engine(tmp_path):
    features = fake_features()
    params = CanonParams(k=4, min_split=2, stability_shuffles=3, members_shown=5)
    engine = Engine()
    result = run_canon(CONFIG, params, features, tmp_path, engine=engine, progress=lambda s: None)
    assert engine.texts[0].startswith("lexical overlap between stem and key: Variant") and len(engine.texts) == 24
    assert result.summary["k"] == 4 and len(result.leaves) == 4 and result.leaves.coherent.all()
    assert len(result.checklist) == 3 and {it["name"] for it in result.checklist} == set(GROUPS)       # the two polarities of family 0 merged
    merged = next(it for it in result.checklist if it["name"] == GROUPS[0])
    assert len(merged["sources"]) == 2 and merged["size"] == 8
    assert result.mapping.canonical.str.match(r"f\d{2}").all() and result.mapping.canonical.nunique() == 3
    assert (result.features.item == result.features.leaf.map({s: it["id"] for it in result.checklist for s in it["sources"]})).all()
    assert (tmp_path / "embeddings.npy").exists() and (tmp_path / "linkage.npy").exists()
    again = run_canon(CONFIG, params, features, tmp_path, engine=engine, progress=lambda s: None)   # embeddings and calls come from the cache
    assert again.summary["checklist"] == result.summary["checklist"]   # (call caching is the backend's, tested in test_llm)


def test_split_of_an_incoherent_cluster(tmp_path):
    features = fake_features()
    params = CanonParams(k=3, min_split=2, stability_shuffles=3, members_shown=5)
    engine = Engine(incoherent={GROUPS[0]})           # the cluster holding both polarities of family 0 is answered as incoherent
    result = run_canon(CONFIG, params, features, tmp_path, engine=engine, progress=lambda s: None)
    assert result.summary["clusters"]["split"] == 1 and len(result.leaves) == 4
    assert sorted(result.leaves.leaf[result.leaves.cluster != result.leaves.leaf].str.len()) == [4, 4]   # two children c<k>.1 and c<k>.2
    assert len(result.checklist) == 3 and {it["name"] for it in result.checklist} == set(GROUPS)
