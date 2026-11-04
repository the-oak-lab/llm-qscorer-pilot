"""Stage 3.2 on one dataset: score every question against the checklist, one question per call, `samples` calls each.

Run:  uv run python scripts/run_presence.py --dataset spacing-exp4 [--checklist results/spacing-exp4/canon/checklist.json] [--samples 2] [--limit 8]
          [--backend gemini|openrouter] [--base-url URL] [--model gemini-3.8-flash] [--thinking medium] [--concurrency 32] [--seed 0] [--out results] [--tag NAME] [--dry-run]
Out:  results/<dataset>/presence[-tag]/responses.jsonl (raw, resumable), samples.csv (one row per call), scores.csv (one row per question,
      share of calls saying present per item), summary.json (tokens, per-item prevalence and agreement), prompt-example.txt
The checklist of another dataset can be scored (--checklist), which is how a held-out dataset is scored for Stage 3.4.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import PresenceParams, load_dataset_config
from qscorer.data import load_questions
from qscorer.presence import build_jobs, run_presence, select_questions


def main():
    d = PresenceParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checklist", default=None, help="checklist.json (default results/<dataset>/canon/checklist.json)")
    ap.add_argument("--samples", type=int, default=d.samples)
    ap.add_argument("--limit", type=int, default=None, help="score only this many questions, spread over the item sets (pilots)")
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default=d.backend)
    ap.add_argument("--base-url", default=d.base_url)
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], default=d.thinking)
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and the call count without calling")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    checklist = json.load(open(args.checklist or root / "canon" / "checklist.json"))
    params = PresenceParams(backend=args.backend, base_url=args.base_url, model=args.model, thinking=args.thinking, samples=args.samples,
                            concurrency=args.concurrency, limit=args.limit, seed=args.seed)
    questions = load_questions(config.questions)
    slopes_file = root / "slopes" / "slopes.csv"
    sets = pd.read_csv(slopes_file).set_index("id")["set"] if slopes_file.exists() else None
    ids = select_questions(questions, config.exclude, params.limit, sets, params.seed)
    if args.dry_run:
        jobs = build_jobs(questions, ids, checklist, config.prompt.context, config.prompt.system, params)
        print(jobs[0].prompt); print(f"\n{len(jobs)} calls for {len(ids)} questions, {len(checklist)} items")
        return
    out = root / ("presence" + (f"-{args.tag}" if args.tag else ""))
    result = run_presence(questions, ids, checklist, config.prompt.context, config.prompt.system, params, out)
    result.samples.to_csv(out / "samples.csv", index=False)
    result.scores.to_csv(out / "scores.csv", index=False)
    summary = result.summary | {"dataset": config.name, "checklist": str(args.checklist or root / "canon" / "checklist.json")}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    (out / "prompt-example.txt").write_text(result.example)
    s = result.summary
    print(f"{config.name}: {s['answered']} of {s['calls']} calls answered for {s['questions_asked']} questions; tokens per call {s['tokens_per_call']}")
    a = pd.DataFrame(s["agreement"]).T
    pd.set_option("display.width", 200)
    print(a.round(3).to_string())
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
