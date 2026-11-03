"""Stage 1 on one dataset: the model's direct learning predictions, one question per call.

Run:  uv run python scripts/run_baseline.py --dataset spacing-exp4 [--limit 8] [--samples 5] [--prompts improve known]
          [--backend gemini|openrouter] [--base-url URL] [--model gemini-3.8-flash] [--temperature 1.0] [--thinking medium] [--no-thoughts] [--concurrency 8] [--passes 3]
          [--seed 0] [--out results] [--tag NAME] [--dry-run]
Out:  results/<dataset>/baseline[-tag]/responses.jsonl (raw, resumable), samples.csv, ratings.csv, evaluation.json,
      summary.json, prompt-example.txt
A pilot: --limit 8 --tag pilot. --dry-run prints the prompts and the call count without calling.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.baseline import build_jobs, run_baseline, select_questions
from qscorer.config import BaselineParams, load_dataset_config
from qscorer.data import load_questions


def main():
    d = BaselineParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None, help="ask only this many questions, spread over the item sets (pilots)")
    ap.add_argument("--samples", type=int, default=d.samples)
    ap.add_argument("--prompts", nargs="+", choices=["improve", "known"], default=d.prompts)
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default=d.backend, help="gemini (GEMINI_API_KEY) or openrouter (OPENROUTER_API_KEY)")
    ap.add_argument("--base-url", default=d.base_url, help="openrouter only: another server of the OpenAI format")
    ap.add_argument("--model", default=d.model, help="model id on the backend")
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], default=d.thinking)
    ap.add_argument("--no-thoughts", action="store_true", help="do not ask for thought summaries")
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--passes", type=int, default=d.passes)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    params = BaselineParams(backend=args.backend, base_url=args.base_url, model=args.model, prompts=args.prompts, samples=args.samples, temperature=args.temperature, thinking=args.thinking,
                            thoughts=not args.no_thoughts, concurrency=args.concurrency, passes=args.passes, limit=args.limit, seed=args.seed)
    root = Path(args.out) / config.name
    slopes = pd.read_csv(root / "slopes" / "slopes.csv") if (root / "slopes" / "slopes.csv").exists() else None
    pairs_q = pd.read_csv(root / "pairs" / "questions.csv") if (root / "pairs" / "questions.csv").exists() else None
    out = root / ("baseline" + (f"-{args.tag}" if args.tag else ""))

    if args.dry_run:
        questions = load_questions(config.questions)
        ids = select_questions(questions, config, params, slopes.set_index("id").set if slopes is not None else None)
        jobs = build_jobs(questions, ids, config, params)
        print(f"{config.name}: {len(ids)} questions x {len(params.prompts)} prompts x {params.samples} samples = {len(jobs)} calls to {params.model} on {params.backend}; "
              f"would write {out}/")
        print(f"\n--- first prompt ({jobs[0].id}) ---\n{jobs[0].prompt}\n--- settings ---\n{json.dumps(jobs[0].settings())}")
        return

    result = run_baseline(config, params, out, slopes, pairs_q)
    result.samples.to_csv(out / "samples.csv", index=False)
    result.ratings.to_csv(out / "ratings.csv", index=False)
    with open(out / "evaluation.json", "w") as f:
        json.dump(result.evaluation, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2)
    (out / "prompt-example.txt").write_text(result.example)

    s = result.summary
    print(f"{config.name}: {s['questions_asked']} questions, {s['calls']} calls, {s['answered']} answered, {s['failed']} failed {s['errors'] or ''}")
    print(f"  tokens per call: " + ", ".join(f"{k.replace('_tokens', '')} {v:.0f}" for k, v in s["tokens_per_call"].items()) + f"; total {sum(s['tokens'].values()):,}")
    for task, a in s["answers"].items():
        c = (s["consistency"] or {}).get(task)
        line = f"  {task}: mean answer {a['mean']:.1f}, sd between questions {a['sd_between_questions']:.1f}"
        if c:
            line += (f"; ICC of one answer {c['icc_single']:.2f}, of the mean over {c['samples_per_question']:.0f} samples {c['reliability_of_mean']:.2f}, "
                     f"samples needed for 0.90: {c['samples_for_0.90']}")
        print(line)
    for task, e in result.evaluation.items():
        if isinstance(e, dict) and "spearman" in e:
            print(f"  {task} vs slope: Spearman {e['spearman']:.2f} [{e['spearman_ci'][0]:.2f}, {e['spearman_ci'][1]:.2f}] over {e['questions']} questions"
                  + (f", within sets {e['within_sets_weighted']:.2f}" if e.get("within_sets_weighted") is not None else "")
                  + (f", AUC high vs low {e['auc_high_vs_low']:.2f}" if "auc_high_vs_low" in e else "")
                  + (f", high member rated higher in {e['high_member_rated_higher']:.0%} of {e['kept_pairs_with_both_rated']} kept pairs" if "high_member_rated_higher" in e else ""))
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
