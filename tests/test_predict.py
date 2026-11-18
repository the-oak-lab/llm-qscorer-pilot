"""The ridge predictor on synthetic data with planted effects: the fit recovers the signs, cross-validation finds signal, apply reproduces
the fit, and the model round-trips through JSON."""
import json

import numpy as np
import pandas as pd

from qscorer.config import PredictParams
from qscorer.predict import apply, evaluate, feature_table, fit

ITEMS = [{"id": f"f{i:02d}", "name": f"item {i}", "definition": "d", "criterion": "c", "part": "stem"} for i in range(1, 9)]


def synthetic(n=240, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 8)) < 0.4).astype(float)
    sets = np.repeat([f"set-{i}" for i in range(4)], n // 4)
    set_effect = pd.Series(sets).map({"set-0": 0.3, "set-1": -0.2, "set-2": 0.1, "set-3": 0.0}).to_numpy()
    slope = shift + set_effect + 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.15 * rng.normal(size=n)
    sd = 0.1 + 0.1 * rng.random(n)
    scores = pd.DataFrame(X, columns=[it["id"] for it in ITEMS]).assign(id=[f"q{seed}-{i}" for i in range(n)])
    slopes = pd.DataFrame({"id": scores.id, "set": sets, "slope": slope, "slope_sd": sd, "p_positive": 1 / (1 + np.exp(-slope / 0.15))})
    return scores, slopes


def test_fit_apply_and_evaluate(tmp_path):
    scores, slopes = synthetic()
    params = PredictParams(alphas=[0.1, 1.0, 10.0, 100.0])
    claimed = pd.DataFrame({"claimed_good_share": {"f01": 0.9, "f02": 0.8, "f04": 0.05, "f05": 0.2, "f06": 0.0},
                            "bad_batches": {"f01": 3, "f02": 5, "f04": 12, "f05": 20, "f06": 4}})   # flaws need share <= 0.1 and >= 10 batches: f04 only
    r = fit(scores, slopes, ITEMS, params, {"model": "fake", "thinking": "low", "temperature": 1.0, "samples": 1, "backend": "fake", "context": "c", "system": "s"}, "synthetic", claimed, boot=100)
    coef = r.model["coefficients"]
    ft = feature_table(r.model)
    assert ft.id.tolist()[0] == "f01" and ft.id.tolist()[-1] == "f02" and ft.direction.tolist()[0] == "more learning" and ft.direction.tolist()[-1] == "less learning"
    assert ft.set_index("id").sign_stability[["f01", "f02"]].min() > 0.95 and ft.sign_stability.between(0, 1).all()
    assert ft.set_index("id").claimed_good_share["f01"] == 0.9 and pd.isna(ft.set_index("id").claimed_good_share["f03"]) and ft.set_index("id").bad_batches["f06"] == 4
    assert coef["f01"] > 0.25 and coef["f02"] < -0.2 and all(abs(coef[f"f{i:02d}"]) < 0.12 for i in range(3, 9))
    assert r.evaluation["cross_validated"]["cv_within_sets_spearman"] > 0.6 and r.evaluation["cross_validated"]["cv_auc_high_vs_low"] > 0.8
    assert len(r.cv) == 4 and r.model["alpha"] in params.alphas
    cvp = r.predictions
    within = np.mean([pd.Series(g.cv_predicted_slope).corr(g.slope, method="spearman") * len(g) for _, g in cvp.groupby("set")]) * 4 / len(cvp)
    assert abs(within - r.evaluation["cross_validated"]["cv_within_sets_spearman"]) < 1e-9   # the held-out column is what CV scored
    pred = apply(r.model, scores)
    assert np.allclose(pred.predicted_slope, r.predictions.predicted_slope) and pred.columns.tolist() == ["id", "predicted_slope", "label", "flaws"]
    assert r.model["flaws"] == ["f04"] and ft.set_index("id").flaw.tolist() == [i == "f04" for i in ft.id]
    assert abs((pred.label == "low improvement expected").mean() - 0.25) < 0.02 and set(pred.label) == {"low improvement expected", "improvement expected"}
    assert (pred.flaws == np.where(scores.f04 == 1, "f04 item 4", "")).all()
    assert abs(pred.predicted_slope.mean() - np.average(slopes.slope, weights=1 / slopes.slope_sd ** 2)) < 0.05   # the pooled scale
    model = json.loads(json.dumps(r.model))                                      # serialisable
    test_scores, test_slopes = synthetic(seed=1, shift=0.5)
    e = evaluate(model, test_scores, test_slopes, params)
    assert e["questions"] == 240 and e["spearman"] > 0.6 and e["within_sets_weighted"] > 0.6 and e["auc_high_vs_low"] > 0.8
    assert len(e["spearman_ci"]) == 2 and e["spearman_ci"][0] < e["spearman"] < e["spearman_ci"][1]
