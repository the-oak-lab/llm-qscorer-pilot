"""Compare Jev's held-out predictions with the researchers' Gemini held-out predictions (checklist_cv) on their 341 questions.
Run from the repo root after scripts/fit_jev_model.py: python scripts/compare_researchers.py
The shuffled-order run is optional (results/nsfv4/jev-tuned-all, the tuned request on datasets/nsfv4/questions.jsonl).
Truth: the researchers' slopes and labels (datasets/nsfv4/researchers/questions-exp4.csv). Every prediction is leave-one-set-out
and demeaned within set before comparing (checklist_cv is a within-set prediction). Paired bootstrap over questions for the
difference between Jev and Gemini."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qscorer.config import PredictParams
from qscorer.predict import auc, demean_within, fit, ridge

model = json.loads(Path("models/spacing-exp4/model.json").read_text())
items = model["items"]; ids_ = [i["id"] for i in items]
R = pd.read_csv("datasets/nsfv4/researchers/questions-exp4.csv").set_index("id")
slopes = R[["set", "slope", "slope_sd", "p_positive"]].reset_index()
params = PredictParams()
sets = R["set"].to_numpy(); y = R.slope.to_numpy(); lab = R.label.fillna("").to_numpy(); low = lab == "low"


def loso(scores: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Held-out predictions at the penalty the scorer's own leave-one-set-out CV picks (the repo's fit)."""
    s = scores.reindex(R.index)
    have = s[ids_].notna().all(axis=1).to_numpy()   # questions this scorer answered; the others get no prediction
    r = fit(s[have].reset_index().rename(columns={"index": "id"}), slopes, items, params, {}, "x", boot=10)
    alpha = r.model["alpha"]
    X = s[ids_].to_numpy(float)
    w = 1 / np.maximum(R.slope_sd.to_numpy(float), 1e-6) ** 2; w = w / w.mean()
    pred = np.full(len(y), np.nan)
    for st in np.unique(sets):
        tr, te = (sets != st) & have, (sets == st) & have
        beta = ridge(demean_within(X[tr], sets[tr]), demean_within(y[tr][:, None], sets[tr])[:, 0], w[tr], alpha)
        pred[te] = X[te] @ beta
    return pred, alpha


def metrics(p: np.ndarray, idx=None) -> dict:
    idx = np.arange(len(p)) if idx is None else idx
    pp, yy, ss, ll, lo = p[idx], y[idx], sets[idx], lab[idx], low[idx]
    pc = demean_within(pp[:, None], ss)[:, 0]
    within = [(spearmanr(pc[ss == s], yy[ss == s])[0], (ss == s).sum()) for s in np.unique(ss)]
    out = {"within-set spearman": sum(a * n for a, n in within) / sum(n for _, n in within), "spearman": spearmanr(pc, yy)[0],
           "AUC high vs low": auc(pc, ll)}
    for frac in (0.10, 0.20):
        k = round(frac * len(pc)); out[f"bottom {int(frac * 100)}% low"] = lo[np.argsort(pc)[:k]].mean()
    return out


L = lambda p: pd.read_csv(p).set_index("id")
jo = L("results/nsfv4/jev-tuned-original-order/scores.csv")
SH = Path("results/nsfv4/jev-tuned-all/scores.csv")
jsh = L(SH) if SH.exists() else None
raw = lambda df: df[[f"{f}.joined" for f in ids_]].set_axis(ids_, axis=1)
preds = {"Gemini checklist_cv (researchers)": R.checklist_cv.to_numpy(),
         "stage1_improve (researchers' baseline)": R.stage1_improve.to_numpy(),
         "stage1_known (researchers' baseline)": R.stage1_known.to_numpy()}
alphas = {}
runs = [("Jev raw, original order", raw(jo)), ("Jev Gemini-calibrated, original order", jo[ids_])]
if jsh is not None:
    runs.append(("Jev raw, shuffled order", raw(jsh)))
for name, df in runs:
    preds[name], alphas[name] = loso(df)
ok = ~np.isnan(preds["Jev raw, shuffled order"]) if jsh is not None else np.ones(len(y), bool)   # the shuffled run lacks the 2 image questions
rows = {n: metrics(p) if not np.isnan(p).any() else metrics(p, np.where(ok)[0]) | {"note": "339 questions"} for n, p in preds.items()}
rows["Gemini in-sample (shipped model)"] = metrics(R.checklist_in_sample.to_numpy())
pd.set_option("display.width", 240)
D = pd.DataFrame(rows).T
D["alpha"] = pd.Series(alphas)
print(f"{len(R)} questions: {int((lab == 'high').sum())} high, {int(low.sum())} low (base rate low {low.mean():.2f})\n")
print(D.to_string(float_format=lambda v: f"{v:.3f}"))

# paired bootstrap: Jev raw (original order) minus Gemini checklist_cv, resampling questions within set
rng = np.random.default_rng(0)
a, g = preds["Jev raw, original order"], preds["Gemini checklist_cv (researchers)"]
diffs = {k: [] for k in ("within-set spearman", "AUC high vs low", "bottom 10% low")}
for _ in range(2000):
    idx = np.concatenate([rng.choice(np.where(sets == s)[0], (sets == s).sum()) for s in np.unique(sets)])
    ma, mg = metrics(a, idx), metrics(g, idx)
    for k in diffs: diffs[k].append(ma[k] - mg[k])
print("\nJev raw (original order) minus Gemini checklist_cv, paired bootstrap 95% CI:")
for k, v in diffs.items():
    v = np.array(v); print(f"  {k}: {np.mean(v):+.3f} ({np.percentile(v, 2.5):+.3f} to {np.percentile(v, 97.5):+.3f}); P(Jev better) {np.mean(v > 0):.2f}")
both = np.where(ok)[0]
if jsh is not None:
    po = raw(jo).reindex(R.index).to_numpy()[both]; ps = raw(jsh).reindex(R.index).to_numpy()[both]
    print(f"\noption order effect on Jev's raw probabilities ({len(both)} questions): mean |diff| {np.abs(po - ps).mean():.3f}; "
          f"correlation of held-out predictions {np.corrcoef(preds['Jev raw, original order'][both], preds['Jev raw, shuffled order'][both])[0, 1]:.3f}")
print(f"Jev raw vs Gemini checklist_cv held-out predictions: spearman {spearmanr(a, g)[0]:.3f}")
D.to_csv("results/nsfv4/compare-researchers.csv")
