"""Scorer comparison without the network: identical scorers agree fully; a scorer that flips one feature shows it."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qscorer.compare import compare, kappa
from qscorer.scorers import read_log

MODEL = json.loads((Path(__file__).resolve().parents[1] / "models" / "spacing-exp4" / "model.json").read_text())
ITEMS = [it["id"] for it in MODEL["items"]]


def table(rng, n=40):
    return pd.DataFrame({"id": [f"q{i}" for i in range(n)], "n_ok": 1} | {f: rng.integers(0, 2, n).astype(float) for f in ITEMS})


def test_identical_scorers():
    a = table(np.random.default_rng(0))
    q, f, s = compare(MODEL, a, a.copy(), "gemini-flash", "jev")
    assert s["questions_compared"] == 40 and q.same_class.all() and (q.slope_gap == 0).all()
    assert s["predictions"]["spearman"] > 0.999 and s["features"]["mean_agreement"] == 1.0
    assert (f.mean_slope_effect == 0).all() and list(q.columns[:3]) == ["id", "slope_gemini-flash", "slope_jev"]


def test_one_feature_flipped():
    a = table(np.random.default_rng(1))
    big = max(ITEMS, key=lambda i: abs(MODEL["coefficients"][i]))
    b = a.copy(); b[big] = 1 - b[big]
    b = b[b.id != "q0"]                              # the candidate failed one question
    q, f, s = compare(MODEL, a, b)
    assert s["questions_compared"] == 39 and s["only_in_reference"] == ["q0"]
    assert f.iloc[0].feature == big and f.iloc[0].agreement == 0.0 and f.iloc[0].kappa < 0
    assert (q.features_disagreeing == 1).all() and (q.disagreeing == big).all()
    assert np.allclose(q.slope_gap.abs(), abs(MODEL["coefficients"][big]))


def test_kappa():
    x = np.array([1, 1, 0, 0.0]); assert kappa(x, x) == 1.0
    assert kappa(np.ones(4), np.ones(4)) is None


def test_read_log(tmp_path):
    log = tmp_path / "responses.jsonl"
    recs = [{"id": "q1|0", "model": "google/gemini-3.8-flash", "usage": {"cost": 0.01}},
            {"id": "q2|0", "model": "anthropic/x", "usage": {"cost": 0.02}},
            {"id": "q3|0", "error": "HTTP 500"}]
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    errors, served, cost = read_log(log)
    assert errors == {"q3": "HTTP 500"} and served == {"google/gemini-3.8-flash": 1, "anthropic/x": 1} and abs(cost - 0.03) < 1e-9
