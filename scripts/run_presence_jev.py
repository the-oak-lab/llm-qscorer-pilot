"""Stage 3.2 with Jev on one dataset: score every question against the checklist, one Jev request per question.

Run:  uv run python scripts/run_presence_jev.py --dataset spacing-exp4 [--checklist results/spacing-exp4/canon/checklist.json]
          [--rewordings configs/jev-rewordings.json] [--api direct|openrouter] [--limit 8] [--concurrency 16] [--out results] [--tag NAME] [--dry-run]
Out:  results/<dataset>/presence-jev[-tag]/responses.jsonl (raw, resumable), samples.csv (one row per request), scores.csv (one row per
      question: Jev's probability per item), summary.json (tokens, the model that answered), request-example.json
Keys: TYPESAFE_API_KEY (--api direct, the default) or OPENROUTER_API_KEY (--api openrouter), in the environment or a .env file.
The checklist of another dataset can be scored (--checklist), which is how a held-out dataset is scored for Stage 3.4.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import load_dataset_config
from qscorer.data import load_questions
from qscorer.jev import REWORDINGS, build_jobs, load_rewordings, run_jev
from qscorer.presence import select_questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checklist", default=None, help="checklist.json (default results/<dataset>/canon/checklist.json)")
    ap.add_argument("--rewordings", default=str(REWORDINGS), help="items asked in other words")
    ap.add_argument("--api", choices=["direct", "openrouter"], default="direct", help="TypeSafe's API (direct) or OpenRouter's")
    ap.add_argument("--limit", type=int, default=None, help="score only this many questions, spread over the item sets (pilots)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the first request without calling")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    checklist_file = args.checklist or root / "canon" / "checklist.json"
    checklist = json.load(open(checklist_file))
    rewordings = load_rewordings(Path(args.rewordings))
    questions = load_questions(config.questions)
    slopes_file = root / "slopes" / "slopes.csv"
    sets = pd.read_csv(slopes_file).set_index("id")["set"] if slopes_file.exists() else None
    ids = select_questions(questions, config.exclude, args.limit, sets, 0)
    if args.dry_run:
        print(json.dumps(json.loads(build_jobs(questions, ids[:1], checklist, config.prompt.context, rewordings, "jev")[0].prompt), indent=2))
        print(f"\n{len(ids)} requests, {len(checklist)} items each")
        return
    out = root / ("presence-jev" + (f"-{args.tag}" if args.tag else ""))
    result = run_jev(questions, ids, checklist, config.prompt.context, rewordings, out, api=args.api, concurrency=args.concurrency)
    result.samples.to_csv(out / "samples.csv", index=False)
    result.scores.to_csv(out / "scores.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary | {"dataset": config.name, "checklist": str(checklist_file)}, f, indent=2)
    (out / "request-example.json").write_text(json.dumps(json.loads(result.example), indent=2) if result.example else "")
    s = result.summary
    print(f"{config.name}: {s['answered']} of {s['calls']} requests answered; served by {s['served_by']}; tokens {s['tokens']}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
