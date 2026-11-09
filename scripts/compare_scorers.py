"""Score the same questions with two scorers and compare their features and predictions.

Run:  python scripts/compare_scorers.py --questions bank.jsonl --out compare-jev [--n 100] [--reference gemini-flash] [--candidate jev]
          [--model models/spacing-exp4/model.json] [--samples 1] [--concurrency 16] [--seed 0] [--context "..."]
          [--jev-variant statement|statement+definition|split] [--reference-seed 0] [--candidate-seed 0]
      Gemini against itself (the noise floor): --candidate gemini-flash --candidate-seed 1.
In:   a question JSONL in the project's format. `--n` questions are drawn from it at random (with `--seed`); all of them when it has fewer.
Out:  <out>/<run>/responses.jsonl and scores.csv for each run (gemini-flash, gemini-flash-seed<n>, jev-<variant>) (resumable: running again
      only sends the missing calls). In <out>/<reference>-vs-<candidate>/: questions.csv (predictions side by side, with the
      features the scorers disagree on), features.csv (agreement and Cohen's kappa per feature, sorted by how much the
      disagreement moves the prediction), features-cut-0.5.csv and features-cut-0.7.csv (Jev's probabilities cut at that
      value), summary.json.
Keys: OPENROUTER_API_KEY in the environment or a .env file. Cost: about two cents per question for gemini-flash; jev costs
      about $0.042 per million input tokens (about $0.0001 per question).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from qscorer.compare import compare
from qscorer.data import load_questions
from qscorer.jev import DEFAULT_VARIANT, VARIANTS
from qscorer.scorers import SCORERS, read_log, score


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
    ap.add_argument("--jev-variant", choices=VARIANTS, default=DEFAULT_VARIANT, help="how each feature is asked of Jev (see qscorer/jev.py)")
    ap.add_argument("--reference-seed", type=int, default=0, help="gemini-flash call seed of the reference (an independent repeat: another seed)")
    ap.add_argument("--candidate-seed", type=int, default=0, help="gemini-flash call seed of the candidate")
    args = ap.parse_args()

    def label(scorer, seed):
        if scorer == "jev":
            return f"jev-{args.jev_variant}"
        return scorer if seed == 0 else f"{scorer}-seed{seed}"
    sides = [(args.reference, args.reference_seed), (args.candidate, args.candidate_seed)]
    a, b = (label(*x) for x in sides)
    if a == b:
        ap.error("the reference and the candidate are the same run: change the scorer or the seed")

    model = json.load(open(args.model))
    questions = load_questions(Path(args.questions))
    ids = list(questions)
    if args.n < len(ids):
        ids = sorted(np.random.default_rng(args.seed).choice(ids, args.n, replace=False).tolist(), key=ids.index)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")

    runs = {}
    for (scorer, seed), name in zip(sides, (a, b)):
        print(f"== {name} ({SCORERS[scorer]}), {len(ids)} questions")
        folder = out / name
        result = score(scorer, questions, ids, model, args.context, folder, samples=args.samples, concurrency=args.concurrency,
                       jev_variant=args.jev_variant, call_seed=seed,
                       timeout=60.0 if scorer == "jev" else 300.0, passes=3)
        result.scores.to_csv(folder / "scores.csv", index=False)
        errors, served, cost = read_log(folder / "responses.jsonl")
        ok = result.samples[result.samples.ok]
        runs[name] = {"scores": result.scores, "model_id": SCORERS[scorer], "calls": result.summary["calls"],
                        "answered": result.summary["answered"], "failed_questions": sorted(set(ids) - set(result.scores.id)),
                        "errors": dict(list(errors.items())[:10]), "served_by": dict(served), "cost_usd": round(cost, 4),
                        "tokens_per_call": result.summary["tokens_per_call"],
                        "elapsed_per_call_s": None}
        lat = [json.loads(line).get("elapsed") for line in (folder / "responses.jsonl").read_text().splitlines() if line.strip()]
        lat = [x for x in lat if x is not None]
        runs[name]["elapsed_per_call_s"] = {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90))} if lat else None
        print(f"   {result.summary['answered']} of {result.summary['calls']} calls answered; served by {dict(served)}; cost ${cost:.4f}"
              f"; {len(ok)} usable")

    q, f, s = compare(model, runs[a]["scores"], runs[b]["scores"], a, b)
    tag = out / f"{a}-vs-{b}"
    tag.mkdir(exist_ok=True)
    q.to_csv(tag / "questions.csv", index=False)
    f.to_csv(tag / "features.csv", index=False)
    # Jev's scores are probabilities; Gemini's are 0 or 1 with one sample. The same comparison with Jev cut at 0.5
    # separates disagreement in the judgments from the difference in scale.
    items = [it["id"] for it in model["items"]]
    s["thresholded_candidate"] = {}
    for t in (0.5, 0.7):
        cut = runs[b]["scores"].copy()
        cut[items] = (cut[items] >= t).astype(float)
        _, f_cut, s_cut = compare(model, runs[a]["scores"], cut, a, f"{b}>={t}")
        f_cut.to_csv(tag / f"features-cut-{t}.csv", index=False)
        s["thresholded_candidate"][str(t)] = {"predictions": s_cut["predictions"], "features": s_cut["features"]}
    s["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in runs.items()}
    s["model"] = args.model
    if b == "jev-split":
        # the joined questions of the same requests: the split and the joined readings side by side
        joined = runs[b]["scores"].copy()
        for c in [c for c in joined.columns if c.endswith(".joined")]:
            joined[c.split(".")[0]] = joined[c]
        jq, jf, js = compare(model, runs[a]["scores"], joined, a, "jev-joined")
        jdir = out / f"{a}-vs-jev-split-joined"
        jdir.mkdir(exist_ok=True)
        jq.to_csv(jdir / "questions.csv", index=False); jf.to_csv(jdir / "features.csv", index=False)
        s["joined_in_same_requests"] = {"predictions": js["predictions"], "features": js["features"]}
        for t in (0.5, 0.7):
            cut = joined.copy(); cut[items] = (cut[items] >= t).astype(float)
            _, _, sc = compare(model, runs[a]["scores"], cut, a, f"jev-joined>={t}")
            s["joined_in_same_requests"][f"cut {t}"] = {"predictions": sc["predictions"], "features": sc["features"]}
    s["jev_variant"] = args.jev_variant
    with open(tag / "summary.json", "w") as fh:
        json.dump(s, fh, indent=2)

    p, ft = s["predictions"], s["features"]
    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"
    print(f"\n{s['questions_compared']} questions scored by both")
    print(f"predicted slope: spearman {fmt(p['spearman'])}, pearson {fmt(p['pearson'])}, mean |gap| {fmt(p['mean_abs_slope_gap'])} "
          f"(training sd {fmt(p['training_slope_sd'])}), same class {fmt(p['same_class'])}, mean |percentile gap| {fmt(p['mean_abs_percentile_gap'])}")
    print("lowest-ranked overlap: " + ", ".join(f"bottom {k} {v['both']}/{v['of']} (chance {v['chance']})" for k, v in p["bottom_overlap"].items()))
    print(f"features: mean agreement {fmt(ft['mean_agreement'])}, mean kappa {fmt(ft['mean_kappa'])}, "
          f"{ft['features_with_kappa_below_0.4']} features below kappa 0.4")
    for t, v in s["thresholded_candidate"].items():
        pc, fc = v["predictions"], v["features"]
        bo = pc["bottom_overlap"]["20%"]
        print(f"with {b} cut at {t}: spearman {fmt(pc['spearman'])}, bottom 20% {bo['both']}/{bo['of']}, same class {fmt(pc['same_class'])}, "
              f"mean |gap| {fmt(pc['mean_abs_slope_gap'])}, feature agreement {fmt(fc['mean_agreement'])}, kappa {fmt(fc['mean_kappa'])}")
    if "joined_in_same_requests" in s:
        for k, v in [("probability", s["joined_in_same_requests"])] + [(k, v) for k, v in s["joined_in_same_requests"].items() if k.startswith("cut")]:
            pc, fc = v["predictions"], v["features"]; bo = pc["bottom_overlap"]["20%"]
            print(f"joined questions, same requests, {k}: spearman {fmt(pc['spearman'])}, bottom 20% {bo['both']}/{bo['of']}, "
                  f"same class {fmt(pc['same_class'])}, feature agreement {fmt(fc['mean_agreement'])}, kappa {fmt(fc['mean_kappa'])}")
    print("features whose disagreement moves the prediction most:")
    print(f.head(8)[["feature", "name", "agreement", "kappa", "mean_slope_effect"]].round(3).to_string(index=False))
    print(f"  wrote {tag}/summary.json, questions.csv, features.csv, features-cut-*.csv")


if __name__ == "__main__":
    main()
