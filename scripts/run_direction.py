"""Which way Gemini claimed each checklist item works: judge every raw feature from discovery as the same as its item, its opposite, or neither.

Run:  uv run python scripts/run_direction.py --dataset spacing-exp4 [--canon DIR] [--items f04,f31] [--batch 60] [--concurrency 64]
          [--backend gemini|openrouter] [--base-url URL] [--model gemini-3.8-flash] [--thinking low] [--out results]
In:   results/<dataset>/canon/features.csv (raw features with their item) and checklist.json
Out:  in the canon folder: direction.csv (one row per raw feature: the polarity asked for, the relation to its item, and the resulting claim
      about the item), direction-items.csv (per item: counts and claimed_good_share, which scripts/run_predict.py reads),
      direction-summary.json, direction-responses.jsonl (raw, resumable). With --items, only those items are judged (a pilot), and the
      outputs go to a direction-pilot folder instead.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import CanonParams, load_dataset_config
from qscorer.direction import run_direction


def main():
    d = CanonParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--canon", default=None, help="the Stage 2.3 folder (default results/<dataset>/canon)")
    ap.add_argument("--items", default=None, help="comma-separated item ids to judge (a pilot)")
    ap.add_argument("--batch", type=int, default=60, help="raw features per call")
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default=d.backend)
    ap.add_argument("--base-url", default=d.base_url)
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--thinking", default=d.thinking, choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    canon = Path(args.canon) if args.canon else Path(args.out) / config.name / "canon"
    features = pd.read_csv(canon / "features.csv")
    checklist = json.load(open(canon / "checklist.json"))
    out = canon
    if args.items:
        keep = args.items.split(",")
        checklist = [it for it in checklist if it["id"] in keep]
        out = canon.parent / "direction-pilot"
    params = CanonParams(backend=args.backend, base_url=args.base_url, model=args.model, thinking=args.thinking, concurrency=args.concurrency)
    rows, items, summary = run_direction(features, checklist, params, config.prompt.system, out, args.batch)
    rows.to_csv(out / "direction.csv", index=False)
    items.to_csv(out / "direction-items.csv", index=False)
    with open(out / "direction-summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{config.name}: {summary['answered']} of {summary['calls']} calls answered; relations {summary['relations']}; tokens {summary['tokens']}")
    pd.set_option("display.width", 200)
    print(items.round(3).to_string(index=False))
    print(f"  wrote {out}/direction*.csv")


if __name__ == "__main__":
    main()
