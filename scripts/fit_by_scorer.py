"""Fit the repo's ridge regression (qscorer.predict.fit) on each scorer's own scores against the observed slopes, and compare
the held-out (leave-one-set-out) results. Run from the repo root: python scripts/fit_by_scorer.py
Inputs: the compare_scorers.py runs in results/compare-jev and results/compare-holdout, results/nsfv4/slopes (scripts/run_slopes.py),
and results/nsfv4/jev-tuned-all (the tuned Jev request on datasets/nsfv4/questions.jsonl).
Scorers on the 200 questions that have every scorer (tuning 100 + holdout 100): Gemini (2-run average), Haiku (1 run),
Jev raw (joined probabilities from the tuned request), Jev tuned (Gemini-calibrated). Jev also on all 339.
Held-out metrics, per scorer at the penalty its own cross-validation chose:
  within-set Spearman (the repo's criterion), overall Spearman with a bootstrap 95% CI, AUC of credibly high (P(slope>0) >= 0.95)
  against credibly low (<= 0.70) learning, and precision at the bottom: of the questions with the lowest 10% / 20% held-out
  predictions, the share that are credibly low-learning (base rate given).
The shipped Gemini model applied to its own training questions is listed for scale (in-sample, not held out)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qscorer.config import PredictParams
from qscorer.predict import apply, auc, demean_within, fit, labels, ridge

model = json.loads(Path("models/spacing-exp4/model.json").read_text())
items = model["items"]
ids_ = [i["id"] for i in items]
slopes = pd.read_csv("results/nsfv4/slopes/slopes.csv")
params = PredictParams()
L = lambda p: pd.read_csv(p).set_index("id")
jall = L("results/nsfv4/jev-tuned-all/scores.csv")
jev_raw_all = jall[[f"{f}.joined" for f in ids_]].set_axis(ids_, axis=1)
jev_tuned_all = jall[ids_]


def stack(*paths):
    return pd.concat([L(p)[ids_] for p in paths])


gem = (stack("results/compare-jev/gemini-flash/scores.csv", "results/compare-holdout/gemini-flash/scores.csv")
       + stack("results/compare-jev/gemini-flash-seed1/scores.csv", "results/compare-holdout/gemini-flash-seed1/scores.csv")) / 2
hai = stack("results/compare-jev/haiku/scores.csv", "results/compare-holdout/haiku/scores.csv")
common = sorted(set(gem.index) & set(hai.index) & set(jall.index))


def heldout(scores: pd.DataFrame, alpha: float):
    t = scores[ids_].join(slopes.set_index("id")[["set", "slope", "slope_sd", "p_positive"]], how="inner").dropna(subset=["slope"])
    X, y, sets = t[ids_].to_numpy(float), t.slope.to_numpy(float), t["set"].to_numpy()
    w = 1 / np.maximum(t.slope_sd.to_numpy(float), 1e-6) ** 2; w = w / w.mean()
    pred = np.full(len(y), np.nan)
    for s in np.unique(sets):
        tr, te = sets != s, sets == s
        beta = ridge(demean_within(X[tr], sets[tr]), demean_within(y[tr][:, None], sets[tr])[:, 0], w[tr], alpha)
        pred[te] = X[te] @ beta
    pred_c = demean_within(pred[:, None], sets)[:, 0]   # predictions are within-set effects; compare questions on that scale
    return t, pred_c


def row(name, scores):
    sc = scores.reset_index().rename(columns={"index": "id"})
    r = fit(sc, slopes, items, params, {}, name, boot=200)
    cv = r.cv.set_index("alpha").loc[r.model["alpha"]]
    t, p = heldout(scores, r.model["alpha"])
    y = t.slope.to_numpy(); low = (t.p_positive <= params.low_p).to_numpy(); lab = labels(t, params).to_numpy()
    rng = np.random.default_rng(0)
    bs = [spearmanr(p[i], y[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    out = {"scorer": name, "questions": len(t), "alpha": r.model["alpha"], "CV within-set spearman": cv.cv_within_sets_spearman,
           "held-out spearman": spearmanr(p, y)[0], "95% CI": f"{np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f}",
           "AUC high vs low": auc(p, lab)}
    for frac in (0.10, 0.20):
        k = round(frac * len(p)); sel = np.argsort(p)[:k]
        out[f"low share in bottom {int(frac * 100)}%"] = f"{low[sel].sum()}/{k}"
    out["base rate low"] = f"{low.mean():.2f}"
    return out, r


rows, fits = [], {}
for name, sc in [("Gemini avg (200)", gem.loc[common]), ("Haiku 1 run (200)", hai.loc[common]), ("Jev raw (200)", jev_raw_all.loc[common]),
                 ("Jev tuned (200)", jev_tuned_all.loc[common]), ("Jev raw (339)", jev_raw_all), ("Jev tuned (339)", jev_tuned_all)]:
    o, r = row(name, sc); rows.append(o); fits[name] = r
# the shipped Gemini model on its own training questions (in-sample)
t = gem.loc[common].join(slopes.set_index("id")[["set", "slope", "p_positive"]], how="inner")
ps = apply(model, gem.loc[common].reset_index().rename(columns={"index": "id"})).set_index("id").predicted_slope.loc[t.index].to_numpy()
rows.append({"scorer": "shipped Gemini model, in-sample (200)", "questions": len(t), "alpha": model["alpha"], "held-out spearman": spearmanr(ps, t.slope)[0],
             "AUC high vs low": auc(ps, labels(t, params).to_numpy())})
pd.set_option("display.width", 260)
D = pd.DataFrame(rows).set_index("scorer")
print(D.to_string(float_format=lambda v: f"{v:.3f}"))
D.to_csv("results/nsfv4/fit-by-scorer.csv")
