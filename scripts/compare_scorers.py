"""Score the same questions with two scorers and compare their features and predictions.

Run:  python scripts/compare_scorers.py --questions bank.jsonl --out compare-jev [--n 100] [--reference gemini-flash] [--candidate jev]
          [--model models/spacing-exp4/model.json] [--samples 1] [--concurrency 16] [--seed 0] [--context "..."]
In:   a question JSONL in the project's format. `--n` questions are drawn from it at random (with `--seed`); all of them when it has fewer.
Out:  <out>/<scorer>/responses.jsonl and scores.csv for each scorer (resumable: running again only sends the missing calls),
      <out>/questions.csv (predictions side by side, with the features the scorers disagree on), <out>/features.csv (agreement
      and Cohen's kappa per feature, sorted by how much the disagreement moves the prediction), <out>/summary.json.
Keys: OPENROUTER_API_KEY in the environment or a .env file. Cost: about one cent per question per scorer for gemini-flash; the
      router bills each call at the price of the model it picks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from qscorer.compare import compare
from qscorer.data import load_questions
from qscorer.presence import run_presence
from qscorer.scorers import SCORERS, presence_params, read_log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reference", choices=list(SCORERS), default="gemini-flash")
    ap.add_argument("--candidate", choices=list(SCORERS), default="jev")
    ap.add_argument("--model", default="models/spacing-exp4/model.json")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", default=None)
    args = ap.parse_args()
    if args.reference == args.candidate:
        ap.error("--reference and --candidate must differ")

    model = json.load(open(args.model))
    scoring = model["scoring"]
    questions = load_questions(Path(args.questions))
    ids = list(questions)
    if args.n < len(ids):
        ids = sorted(np.random.default_rng(args.seed).choice(ids, args.n, replace=False).tolist(), key=ids.index)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")

    runs = {}
    for scorer in (args.reference, args.candidate):
        print(f"== {scorer} ({SCORERS[scorer]}), {len(ids)} questions")
        params = presence_params(scoring, scorer, samples=args.samples, concurrency=args.concurrency, timeout=300.0, passes=3)
        folder = out / scorer
        result = run_presence(questions, ids, model["items"], args.context or scoring["context"], scoring["system"], params, folder)
        result.scores.to_csv(folder / "scores.csv", index=False)
        errors, served, cost = read_log(folder / "responses.jsonl")
        ok = result.samples[result.samples.ok]
        runs[scorer] = {"scores": result.scores, "model_id": SCORERS[scorer], "calls": result.summary["calls"],
                        "answered": result.summary["answered"], "failed_questions": sorted(set(ids) - set(result.scores.id)),
                        "errors": dict(list(errors.items())[:10]), "served_by": dict(served), "cost_usd": round(cost, 4),
                        "tokens_per_call": result.summary["tokens_per_call"],
                        "elapsed_per_call_s": None}
        lat = [json.loads(line).get("elapsed") for line in (folder / "responses.jsonl").read_text().splitlines() if line.strip()]
        lat = [x for x in lat if x is not None]
        runs[scorer]["elapsed_per_call_s"] = {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90))} if lat else None
        print(f"   {result.summary['answered']} of {result.summary['calls']} calls answered; served by {dict(served)}; cost ${cost:.4f}"
              f"; {len(ok)} usable")

    a, b = args.reference, args.candidate
    q, f, s = compare(model, runs[a]["scores"], runs[b]["scores"], a, b)
    q.to_csv(out / "questions.csv", index=False)
    f.to_csv(out / "features.csv", index=False)
    s["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in runs.items()}
    s["model"] = args.model
    with open(out / "summary.json", "w") as fh:
        json.dump(s, fh, indent=2)

    p, ft = s["predictions"], s["features"]
    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"
    print(f"\n{s['questions_compared']} questions scored by both")
    print(f"predicted slope: spearman {fmt(p['spearman'])}, pearson {fmt(p['pearson'])}, mean |gap| {fmt(p['mean_abs_slope_gap'])} "
          f"(training sd {fmt(p['training_slope_sd'])}), same class {fmt(p['same_class'])}, mean |percentile gap| {fmt(p['mean_abs_percentile_gap'])}")
    print(f"features: mean agreement {fmt(ft['mean_agreement'])}, mean kappa {fmt(ft['mean_kappa'])}, "
          f"{ft['features_with_kappa_below_0.4']} features below kappa 0.4")
    print("features whose disagreement moves the prediction most:")
    print(f.head(8)[["feature", "name", "agreement", "kappa", "mean_slope_effect"]].round(3).to_string(index=False))
    print(f"  wrote {out}/summary.json, questions.csv, features.csv")


if __name__ == "__main__":
    main()
