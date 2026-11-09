"""Test alternative Jev wordings of checklist features in one request per question, against LLM scorers.

Run:  python scripts/test_jev_phrasings.py [--phrasings configs/jev-phrasings.json] [--results results/compare-jev]
          [--questions datasets/nsfv4/demo-100.jsonl] [--target gemini-flash gemini-flash-seed1] [--check haiku haiku-seed1]
          [--out results/compare-jev/jev-phrasings] [--folds 5] [--repeats 20]
How:  each request asks every variant of every feature in the file (ids `<feature>.<variant>`) plus two controls per feature:
      `def` (the current statement with the definition as criteria) and `stmt` (the current statement alone). Each variant
      is scored against the share of target runs saying present by
        - AUC: Jev's probability on questions where the target runs agree (threshold-free: does Jev rank them the same way)
        - err/kappa: a threshold tuned on training folds, applied to held-out questions (mean absolute error from the share)
      and the same against the check runs. Only questions present in every run are used.
Out:  <out>/responses.jsonl (resumable), <out>/probs.csv, <out>/variants.csv; the table on stdout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qscorer.data import load_questions
from qscorer.jev import JEV_MODEL, Jev, nouls, questions_for, state, statement
from qscorer.llm import Job

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def auc(p: np.ndarray, share: np.ndarray) -> float:
    pos, neg = p[share == 1], p[share == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a >= 0.5, b >= 0.5
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe >= 1 else float((po - pe) / (1 - pe))


def cv(p: np.ndarray, share: np.ndarray, folds: int, repeats: int, rng) -> tuple[float, float, float]:
    """Held-out error and kappa of a tuned threshold, and the threshold fitted on all questions."""
    def best(pp, yy):
        return min(GRID, key=lambda t: (np.abs((pp >= t) - yy).mean(), abs(t - 0.7)))
    errs, ks = [], []
    for _ in range(repeats):
        fold = rng.permutation(len(p)) % folds
        x = np.zeros(len(p))
        for k in range(folds):
            tr, te = fold != k, fold == k
            x[te] = (p[te] >= best(p[tr], share[tr])).astype(float)
        errs.append(np.abs(x - share).mean()); ks.append(kappa(x, share))
    return float(np.mean(errs)), float(np.nanmean(ks)), float(best(p, share))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrasings", default="configs/jev-phrasings.json")
    ap.add_argument("--results", default="results/compare-jev")
    ap.add_argument("--questions", default="datasets/nsfv4/demo-100.jsonl")
    ap.add_argument("--model", default="models/spacing-exp4/model.json")
    ap.add_argument("--target", nargs="+", default=["gemini-flash", "gemini-flash-seed1"])
    ap.add_argument("--check", nargs="*", default=["haiku", "haiku-seed1"])
    ap.add_argument("--out", default="results/compare-jev/jev-phrasings")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text())
    by_id = {it["id"]: it for it in model["items"]}
    phr = {k: v for k, v in json.loads(Path(args.phrasings).read_text()).items() if not k.startswith("_")}
    R = Path(args.results)
    load = lambda n: pd.read_csv(R / n / "scores.csv").set_index("id")
    target, check = {n: load(n) for n in args.target}, {n: load(n) for n in args.check}
    questions = load_questions(Path(args.questions))
    ids = [q for q in questions if all(q in s.index for s in list(target.values()) + list(check.values()))]

    asked: dict[str, dict] = {}
    for f, variants in phr.items():
        it = by_id[f]
        asked[f"{f}.def"] = questions_for([it], "statement+definition")[f]
        asked[f"{f}.stmt"] = {"type": "noul", "instructions": statement(it["criterion"])}
        for v, text in variants.items():
            asked[f"{f}.{v}"] = {"type": "noul", "instructions": text}
    context = model["scoring"]["context"]
    jobs = [Job(f"{q}|0", json.dumps({"model": JEV_MODEL, "state": state(questions[q], context), "questions": asked}),
                schema=None, thinking=None, thoughts=False) for q in ids]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keys = list(asked)
    records = Jev(JEV_MODEL, concurrency=args.concurrency).many(jobs, out / "responses.jsonl", 3, lambda raw: nouls(raw, keys) is not None)
    probs = pd.DataFrame([{"id": j.id.split("|")[0], **(nouls(records[j.id], keys) or {})} for j in jobs if j.id in records]).set_index("id")
    probs.to_csv(out / "probs.csv")
    cost = sum(float((r.get("usage") or {}).get("cost") or 0) for r in records.values())
    print(f"{len(probs)} of {len(ids)} questions answered, {len(keys)} questions per request, cost ${cost:.4f}\n")

    ids = list(probs.index)
    rng = np.random.default_rng(0)
    rows = []
    for f in phr:
        t_share = np.mean([s.loc[ids, f].to_numpy(float) for s in target.values()], axis=0)
        c_share = np.mean([s.loc[ids, f].to_numpy(float) for s in check.values()], axis=0) if check else None
        for key in [k for k in keys if k.startswith(f + ".")]:
            p = probs[key].to_numpy(float)
            err, k, t = cv(p, t_share, args.folds, args.repeats, rng)
            r = {"feature": f, "variant": key.split(".", 1)[1], "AUC target": auc(p, t_share), "held-out err": err, "held-out kappa": k,
                 "threshold": t, "jev yes at t": float((p >= t).mean()), "target yes": float(t_share.mean())}
            if c_share is not None:
                r["AUC check"] = auc(p, c_share)
                r["kappa check (at t)"] = kappa((p >= t).astype(float), c_share)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(out / "variants.csv", index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = D.loc[D.groupby("feature")["held-out err"].idxmin()][["feature", "variant", "held-out err", "held-out kappa", "AUC target"]]
    ctrl = D[D.variant == "def"].set_index("feature")[["held-out err", "held-out kappa"]].add_prefix("def ")
    print("\nbest variant per feature (lowest held-out error) against the current `def` wording:")
    print(best.set_index("feature").join(ctrl).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
