"""Stage 2.1 on one dataset: the batch plan from the Stage 0.2 pairs.

Run:  uv run python scripts/run_batches.py --dataset spacing-exp4 [--rounds 5] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/batches[-tag]/batches.csv (one row per question in a batch: round, set, kind, batch, position, id,
      pool, pair, accuracy), summary.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.batches import BatchParams, plan_batches
from qscorer.config import load_dataset_config


def main():
    d = BatchParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rounds", type=int, default=d.rounds)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    pairs = pd.read_csv(root / "pairs" / "pairs.csv")
    params = BatchParams(rounds=args.rounds, seed=args.seed)
    plan, summary = plan_batches(pairs, params)
    out = root / ("batches" + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "batches.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    s = summary
    print(f"{config.name}: {s['pairs']} pairs, {s['questions']} questions, {params.rounds} rounds")
    print("  batches per round: " + ", ".join(f"{k} {v:.0f}" for k, v in s["batches_per_round"].items())
          + f"; calls per round with one call per batch and condition {s['calls_per_round_one_per_condition']}, total {s['calls_total_one_per_condition']}")
    print("  per set per round: " + "; ".join(f"{st} " + "/".join(str(v[k]) for k in ("mixed", "good", "bad")) for st, v in s["per_set_per_round"].items()) + " (mixed/good/bad)")
    print("  left out per round: " + "; ".join(f"round {r['round']} " + ", ".join(f"{k} {len(v)}" for k, v in r["left_out"].items()) for r in s["rounds"])
          + f"; most times any unit sits out: {s['left_out_per_question_max']}")
    print(f"  repeated companions over all rounds: {s['repeated_companions_total']}")
    print("  accuracy range within a batch, median / 90% / max: " + "; ".join(f"{k} {v['0.5']:.2f} / {v['0.9']:.2f} / {v['1.0']:.2f}" for k, v in s["accuracy_range_within_batch"].items()))
    print(f"  high members by position in mixed batches: {s['high_members_by_position_in_mixed_batches']}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
