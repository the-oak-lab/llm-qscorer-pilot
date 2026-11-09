"""Holdout table: every scorer against the Gemini 2-run average (floor: seed 1 against seed 0).
Usage: python scripts/holdout_table.py results/compare-holdout models/spacing-exp4/model.json results/nsfv4/slopes/slopes.csv
The results dir holds the compare_scorers.py runs gemini-flash, gemini-flash-seed1, haiku and jev-tuned.
The observed-learning columns are in-sample for Gemini: the shipped model was fitted on these questions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qscorer.predict import auc

R, MODEL, SLOPES = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m = json.loads(MODEL.read_text())
items = [i["id"] for i in m["items"]]
coef = np.array([m["coefficients"][f] for f in items])
L = lambda n: pd.read_csv(R / n / "scores.csv").set_index("id")
g0, g1, hk, jt = L("gemini-flash"), L("gemini-flash-seed1"), L("haiku"), L("jev-tuned")
ids = [q for q in g0.index if all(q in s.index for s in (g1, hk, jt))]
G0, G1 = g0.loc[ids, items].to_numpy(float), g1.loc[ids, items].to_numpy(float)
G = (G0 + G1) / 2
JT = jt.loc[ids, items].to_numpy(float)
JU = (jt.loc[ids, [f"{f}.joined" for f in items]].to_numpy(float) >= 0.7).astype(float)
H = hk.loc[ids, items].to_numpy(float)
obs = pd.read_csv(SLOPES).set_index("id").loc[ids]
low = np.where(obs.p_positive.to_numpy() <= 0.70, "low", "high")
slope = lambda X: m["intercept"] + X @ coef
c10, c20 = np.percentile(m["training_predictions"], [10, 20])


def kappa(a, b):
    a, b = a >= 0.5, b >= 0.5
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return None if pe >= 1 else (po - pe) / (1 - pe)


def met(X, ref):
    s, rs = slope(X), slope(ref)
    ks = [k for j in range(len(items)) if (k := kappa(X[:, j], ref[:, j])) is not None]
    r = {"feature err": np.abs(X - ref).mean(), "kappa": np.mean(ks), "slope spearman": spearmanr(s, rs)[0],
         "slope |gap|": np.abs(s - rs).mean(), "slope gap": (s - rs).mean()}
    for nm, c in (("0.123", c10), ("0.161", c20)):
        a, b = s < c, rs < c
        r[f"<{nm} flagged/shared/ref"] = f"{a.sum()}/{(a & b).sum()}/{b.sum()}"
    r["bottom20 shared"] = len(set(np.argsort(rs)[:20]) & set(np.argsort(s)[:20]))
    r["obs spearman"] = spearmanr(s, obs.slope)[0]
    r["obs AUC low"] = auc(s, low)
    return r


rows = {"Gemini seed 1 vs seed 0 (floor)": met(G1, G0), "Haiku, 1 run": met(H, G), "Jev untuned (joined, cut 0.7)": met(JU, G),
        "Jev tuned (frozen)": met(JT, G)}
ref_obs = {"Gemini 2-run average": {"obs spearman": spearmanr(slope(G), obs.slope)[0], "obs AUC low": auc(slope(G), low)}}
pd.set_option("display.width", 260)
print(f"{len(ids)} questions; {int((low == 'low').sum())} credibly low-learning\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
print("\nGemini itself against observed learning:", {k: round(v, 3) for k, v in ref_obs["Gemini 2-run average"].items()})
