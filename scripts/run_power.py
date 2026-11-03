"""Stage 2.4 power simulation on one dataset's pairs: plans the batches for each number of rounds
given, simulates one feature's attributions under the model in `qscorer/power.py` for each number of
calls per batch and condition, and reports the rejection rates of the planned tests.

Run:  uv run python scripts/run_power.py --dataset spacing-exp4 [--rounds 3 5] [--samples 1 2] [--sims 500] [--base-rate 0.25]
          [--sigma-question 1.0] [--sigma-batch 0.5] [--sigma-call 0.5] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/power[-tag]/power.csv (one row per rounds, samples and scenario), summary.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.batches import BatchParams, plan_batches
from qscorer.config import load_dataset_config
from qscorer.power import TESTS, PowerParams, run_power


def main():
    d = PowerParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sims", type=int, default=d.sims)
    ap.add_argument("--base-rate", type=float, default=d.base_rate)
    ap.add_argument("--sigma-question", type=float, default=d.sigma_question)
    ap.add_argument("--sigma-batch", type=float, default=d.sigma_batch)
    ap.add_argument("--sigma-call", type=float, default=d.sigma_call)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    pairs = pd.read_csv(root / "pairs" / "pairs.csv")
    tables = []
    for rounds in args.rounds:
        plan, _ = plan_batches(pairs, BatchParams(rounds=rounds, seed=args.seed))
        for samples in args.samples:
            params = PowerParams(base_rate=args.base_rate, sigma_question=args.sigma_question, sigma_batch=args.sigma_batch, sigma_call=args.sigma_call,
                                 samples=samples, sims=args.sims, seed=args.seed)
            print(f"{config.name}: {rounds} rounds, {samples} call(s) per batch and condition, {len(plan.pair.unique())} pairs")
            t = run_power(plan, params)
            t.insert(0, "samples", samples)
            t.insert(0, "rounds", rounds)
            tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    out = root / ("power" + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "power.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump({"params": params.model_dump(mode="json") | {"rounds": args.rounds, "samples": args.samples}, "pairs": int(pairs.shape[0])}, f, indent=2)
    pd.set_option("display.width", 200)
    cols = ["rounds", "samples", "delta", "kappa", "rate_high", "rate_labelled_high"] + [f"{t}_0.05" for t in TESTS] + ["quality_0.0125", "quality_mixed_0.0125", "inversion_0.0125", "family_true_discoveries", "family_false_discoveries"]
    print(table[cols].round(2).to_string(index=False))
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
