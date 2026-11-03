"""Stage 2.3 scoring and 2.4 inference on one dataset: every contrast on every canonical feature of a mapping.

Run:  uv run python scripts/run_contrasts.py --dataset spacing-exp4 [--mapping results/spacing-exp4/canon/cut-30.csv] [--discovery DIR]
          [--min-calls 10] [--flips 9999] [--boot 2000] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/contrasts[-tag]/features.csv (side, test, feature, estimate, interval, p, q), omnibus.csv, summary.json
The mapping is a CSV with batch, condition, sample, feature and canonical (blank for none); without one, the raw feature
names (lower-cased, punctuation stripped) are the canonical features, a smoke test of the pipeline.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import load_dataset_config
from qscorer.contrasts import PRIMARY, ContrastParams, analyse, raw_mapping


def main():
    d = ContrastParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mapping", default=None, help="CSV mapping each raw feature to a canonical id; default: the raw names")
    ap.add_argument("--discovery", default=None, help="the Stage 2.2 folder (default results/<dataset>/discovery)")
    ap.add_argument("--min-calls", type=int, default=d.min_calls)
    ap.add_argument("--flips", type=int, default=d.flips)
    ap.add_argument("--boot", type=int, default=d.boot)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    discovery = Path(args.discovery) if args.discovery else root / "discovery"
    plan = pd.read_csv(root / "batches" / "batches.csv")
    calls = pd.read_csv(discovery / "calls.csv")
    features = pd.read_csv(discovery / "features.csv")
    attributions = pd.read_csv(discovery / "attributions.csv")
    mapping = pd.read_csv(args.mapping, dtype={"canonical": str}) if args.mapping else raw_mapping(features)
    params = ContrastParams(min_calls=args.min_calls, flips=args.flips, boot=args.boot, seed=args.seed)
    table, omnibus = analyse(plan, calls, features, attributions, mapping, params)

    out = root / ("contrasts" + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "features.csv", index=False)
    omnibus.to_csv(out / "omnibus.csv", index=False)
    summary = {"dataset": config.name, "params": params.model_dump(mode="json"), "mapping": args.mapping or "raw names", "discovery": str(discovery),
               "raw_features": int(len(features)), "canonical_features_tested": int(table.feature.nunique()) if len(table) else 0,
               "mapped_features": int(mapping.canonical.notna().sum()),
               "fdr_rejections": {f"{s}:{t}": int(g.fdr_rejected.sum()) for (s, t), g in table.groupby(["side", "test"])} if len(table) else {}}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 200)
    print(f"{config.name}: {summary['raw_features']} raw features, {summary['mapped_features']} mapped, {summary['canonical_features_tested']} canonical features "
          f"named in at least {params.min_calls} calls (mapping: {summary['mapping']})")
    print(omnibus.round(4).to_string(index=False))
    for side in ("good", "bad"):
        for test in PRIMARY:
            g = table[(table.side == side) & (table.test == test)].sort_values("p").head(5)
            print(f"\n  {side} {test}: {int(table[(table.side == side) & (table.test == test)].fdr_rejected.sum())} features at FDR 0.05; smallest p:")
            print(g[["feature", "calls", "estimate", "ci_low", "ci_high", "p", "q"]].round(3).to_string(index=False))
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
