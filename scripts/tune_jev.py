"""Tune how Jev's answers become feature scores, feature by feature, against the LLM scorers; no new calls.

Run:  python scripts/tune_jev.py [--results results/compare-jev] [--jev jev-split] [--llm gemini-flash gemini-flash-seed1 haiku haiku-seed1]
          [--model models/spacing-exp4/model.json] [--slopes results/nsfv4/slopes/slopes.csv] [--folds 5] [--repeats 20]
          [--out configs/jev-calibration.json]
In:   a Jev `split` run (each request asks every feature joined and the compound ones split, so both forms come from the same
      requests) and LLM runs of the same questions. The target is the share of LLM runs that said a feature is present, the
      number the regression was fitted on (the share of Gemini calls), with both models weighted by their run count.
How:  per feature, the choices are made on training folds and scored on the held-out questions, in stages:
        S0  joined form, probability cut at 0.7 (the earlier best)
        S1  joined or split form, whichever is closer to the LLM share, cut at 0.7
        S2  form and threshold chosen together (thresholds 0.05 to 0.95)
        S3  as S2, or a calibrated probability: Jev's probability mapped onto the LLM share by a logistic fit
      "Closer" is the mean absolute difference from the LLM share. The noise floor is each LLM run against the other runs.
Out:  the stage table (feature agreement, predicted slope against the LLM-consensus slope, flags below training-percentile
      cutoffs, and observed learning when --slopes is given) and the S3 choices fitted on all questions, written to --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from qscorer.jev import load_splits
from qscorer.predict import auc

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """a, b of y ~ sigmoid(a * logit(p) + b), by cross-entropy with fractional targets."""
    x = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    def loss(w):
        z = np.clip(1 / (1 + np.exp(-(w[0] * x + w[1]))), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z)) + 1e-3 * (w[0] - 1) ** 2
    w = minimize(loss, np.array([1.0, 0.0]), method="L-BFGS-B").x
    return float(w[0]), float(w[1])


def apply_choice(c: dict, forms: dict[str, np.ndarray]) -> np.ndarray:
    p = forms[c["form"]]
    if c["mode"] == "threshold":
        return (p >= c["threshold"]).astype(float)
    x = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    return 1 / (1 + np.exp(-(c["a"] * x + c["b"])))


def choose(stage: str, forms: dict[str, np.ndarray], y: np.ndarray) -> dict:
    cands = []
    names = list(forms) if stage != "S0" else ["joined"]
    for form in names:
        for t in ([0.7] if stage in ("S0", "S1") else GRID):
            c = {"form": form, "mode": "threshold", "threshold": float(t)}
            cands.append((np.abs(apply_choice(c, forms) - y).mean(), abs(t - 0.7), form != "joined", c))
        if stage == "S3":
            a, b = platt(forms[form], y)
            c = {"form": form, "mode": "probability", "a": a, "b": b}
            cands.append((np.abs(apply_choice(c, forms) - y).mean(), 0.0, form != "joined", c))
    return min(cands, key=lambda t: t[:3])[3]   # least error; ties go to the threshold nearest 0.7, then to the joined form


def kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a >= 0.5, b >= 0.5
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return None if pe >= 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/compare-jev")
    ap.add_argument("--jev", default="jev-split")
    ap.add_argument("--llm", nargs="+", default=["gemini-flash", "gemini-flash-seed1", "haiku", "haiku-seed1"])
    ap.add_argument("--model", default="models/spacing-exp4/model.json")
    ap.add_argument("--slopes", default="results/nsfv4/slopes/slopes.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/jev-calibration.json")
    args = ap.parse_args()

    R = Path(args.results)
    m = json.loads(Path(args.model).read_text())
    items = [i["id"] for i in m["items"]]
    names = {i["id"]: i["name"] for i in m["items"]}
    coef = np.array([m["coefficients"][f] for f in items])
    jev = pd.read_csv(R / args.jev / "scores.csv").set_index("id")
    llm = {n: pd.read_csv(R / n / "scores.csv").set_index("id") for n in args.llm}
    ids = [q for q in jev.index if all(q in s.index for s in llm.values())]
    stack = np.stack([llm[n].loc[ids, items].to_numpy(float) for n in args.llm])
    share = stack.mean(axis=0)
    splits = load_splits()
    forms = {}
    for f in items:
        # a `tuned` run keeps the raw probabilities in <id>.joined / <id>.split (its <id> is already calibrated);
        # a `split` run has the split OR in <id> and the joined probability in <id>.joined
        forms[f] = {"joined": jev.loc[ids, f"{f}.joined" if f"{f}.joined" in jev.columns else f].to_numpy(float)}
        if f in splits:
            forms[f]["split"] = jev.loc[ids, f"{f}.split" if f"{f}.split" in jev.columns else f].to_numpy(float)
    slope = lambda X: m["intercept"] + X @ coef
    train = np.sort(m["training_predictions"])
    cutoffs = {"p10": float(np.percentile(train, 10)), "p20": float(np.percentile(train, 20))}
    obs = pd.read_csv(args.slopes).set_index("id").loc[ids] if args.slopes and Path(args.slopes).exists() else None
    cons_slope = slope(share)

    def metrics(X: np.ndarray, ref_share: np.ndarray, ref_slope: np.ndarray) -> dict:
        s = slope(X)
        ks = [k for j in range(len(items)) if (k := kappa(X[:, j], ref_share[:, j])) is not None]
        r = {"feature |err|": float(np.abs(X - ref_share).mean()), "kappa": float(np.mean(ks)),
             "slope spearman": float(spearmanr(s, ref_slope)[0]), "slope |gap|": float(np.abs(s - ref_slope).mean()),
             "slope gap": float((s - ref_slope).mean())}
        for k, c in cutoffs.items():
            a, b = s < c, ref_slope < c
            r[f"flag {k}: n / shared / ref"] = f"{int(a.sum())} / {int((a & b).sum())} / {int(b.sum())}"
        if obs is not None:
            r["obs spearman"] = float(spearmanr(s, obs.slope)[0])
            r["obs AUC low"] = auc(s, np.where(obs.p_positive.to_numpy() <= 0.70, "low", "high"))
        return r

    rows = []
    for r in range(len(args.llm)):   # noise floor: one LLM run against the share of the others
        others = np.delete(stack, r, axis=0).mean(axis=0)
        rows.append({"stage": f"floor: {args.llm[r]} vs other LLM runs"} | metrics(stack[r], others, slope(others)))
    rng = np.random.default_rng(args.seed)
    n = len(ids)
    for stage in ("S0", "S1", "S2", "S3"):
        reps = []
        for _ in range(args.repeats):
            X = np.zeros((n, len(items)))
            fold = rng.permutation(n) % args.folds
            for k in range(args.folds):
                tr, te = fold != k, fold == k
                for j, f in enumerate(items):
                    c = choose(stage, {nm: p[tr] for nm, p in forms[f].items()}, share[tr, j])
                    X[te, j] = apply_choice(c, {nm: p[te] for nm, p in forms[f].items()})
            reps.append(metrics(X, share, cons_slope))
        num = {k: float(np.mean([d[k] for d in reps])) for k in reps[0] if isinstance(reps[0][k], float)}
        txt = {k: reps[0][k] for k in reps[0] if not isinstance(reps[0][k], float)}
        rows.append({"stage": f"{stage} (held-out, mean of {args.repeats})"} | num | {k: v + " (repeat 1)" for k, v in txt.items()})
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20)
    print(f"{n} questions; target: share of {len(args.llm)} LLM runs; cutoffs {cutoffs}\n")
    print(pd.DataFrame(rows).set_index("stage").to_string(float_format=lambda v: f"{v:.3f}"))

    final = {f: choose("S3", forms[f], share[:, j]) for j, f in enumerate(items)}
    out = {"_about": "Per-feature Jev scoring fitted by scripts/tune_jev.py on a `split` run against the share of LLM runs saying present. "
                     "form: joined (the full criterion) or split (OR of the parts in configs/jev-splits.json); mode: threshold "
                     "(present when the probability is at least `threshold`) or probability (sigmoid(a * logit(p) + b)).",
           "fitted_on": {"jev_run": args.jev, "llm_runs": args.llm, "questions": n}, "features": final}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    tab = pd.DataFrame([{"feature": f, "name": names[f][:44], "coef": m["coefficients"][f], **{k: v for k, v in c.items() if k in ("form", "mode", "threshold")}}
                        for f, c in final.items()])
    print(f"\nchoices fitted on all {n} questions (written to {args.out}):")
    print(f"  forms: {tab.form.value_counts().to_dict()}; modes: {tab["mode"].value_counts().to_dict()}")
    print(tab[(tab.form == "split") | (tab["mode"] == "probability") | ((tab["mode"] == "threshold") & (tab.threshold != 0.7))].to_string(index=False))


if __name__ == "__main__":
    main()
