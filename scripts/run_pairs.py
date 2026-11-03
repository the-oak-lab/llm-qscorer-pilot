"""Stage 0.2 and 0.4 on one dataset: high- and low-learning pools, matched pairs with the
separation test, and the selection diagnostics, from the Stage 0.1 table.

Run:  uv run python scripts/run_pairs.py --dataset spacing-exp4 [--slopes results/spacing-exp4/slopes] [--quantile 0.25]
          [--caliper 0.10] [--separation 0.90] [--min-difference 0.20] [--reliability-splits 5] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/pairs/questions.csv, pairs.csv, diagnostics.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import PairParams, SlopeParams, load_dataset_config
from qscorer.pairs import run_pairs


def main():
    defaults = PairParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a name under configs/datasets/ or a path to a config file")
    ap.add_argument("--slopes", default=None, help="the Stage 0.1 folder (default results/<dataset>/slopes)")
    ap.add_argument("--quantile", type=float, default=defaults.quantile)
    ap.add_argument("--caliper", type=float, default=defaults.caliper, help="in share correct, 0 to 1")
    ap.add_argument("--separation", type=float, default=defaults.separation)
    ap.add_argument("--min-difference", type=float, default=defaults.min_difference)
    ap.add_argument("--reliability-splits", type=int, default=defaults.reliability_splits)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--out", default="results", help="results root; files go to <out>/<dataset>/pairs/")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    slopes = Path(args.slopes) if args.slopes else Path(args.out) / config.name / "slopes"
    table = pd.read_csv(slopes / "slopes.csv")
    with open(slopes / "summary.json") as f:
        slope_params = SlopeParams(**json.load(f)["params"])
    params = PairParams(quantile=args.quantile, caliper=args.caliper, separation=args.separation, min_difference=args.min_difference,
                        reliability_splits=args.reliability_splits, seed=args.seed)
    result = run_pairs(table, params, config, slope_params)

    out = Path(args.out) / config.name / ("pairs" + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    result.questions.to_csv(out / "questions.csv", index=False)
    result.pairs.to_csv(out / "pairs.csv", index=False)
    d = dict(result.diagnostics, slopes=str(slopes))
    with open(out / "diagnostics.json", "w") as f:
        json.dump(d, f, indent=2)

    p = d["pairs"]
    print(f"{config.name}: pools per set {{" + ", ".join(f"{s}: {v['high']}/{v['low']}" for s, v in d["pools"].items()) + "}")
    print(f"  pairs: {p['matched']} matched within a caliper of {params.caliper:.2f}, {p['selected']} selected (P >= {params.separation}, difference >= {params.min_difference}); "
          f"{p['selected_at_separation']['0.95']} at P >= 0.95; {p['separated_but_below_min_difference']} separated but below the minimum difference; "
          f"without a caliper {p['without_caliper']['matched']} matched, {p['without_caliper']['selected']} selected")
    print(f"  selected pairs: median slope difference {p['selected_slope_difference_median']:.3f}; accuracy difference quantiles {{" +
          ", ".join(f"{k}: {v:.3f}" for k, v in p["accuracy_difference"].items()) + "}")
    a = d["alternatives"]
    print(f"  alternatives: A {a['A_quantiles_only']}, B {a['B_meaningful_slope']}, C {a['C_sign_probability']}")
    s = d["slope_vs_first_practice_accuracy"]
    print(f"  slope vs first-practice accuracy, Spearman {s['spearman']:.2f}, within sets {{" + ", ".join(f"{k}: {v:.2f}" for k, v in s["within_sets"].items()) + "}")
    if "replication" in d:
        m = d["replication"]["median"]
        print(f"  split-half replication ({params.reliability_splits} splits): three-way label kappa {m['kappa_three_way']:.2f}, high-vs-low kappa {m['kappa_high_vs_low']:.2f} "
              f"over {m['extreme_in_both']:.0f} questions extreme in both halves, {m['high_low_flips']:.0f} high-low flips; "
              f"{m['pairs_selected_per_half']:.0f} pairs selected per half, direction replicates in the other half {m['direction_replicates']:.0%}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
