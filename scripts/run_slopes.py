"""Stage 0.1 on one dataset: estimate every question's practice slope.

Run:  uv run python scripts/run_slopes.py --dataset spacing-exp4 [--max-opportunities 7] [--min-support 10]
          [--timeouts keep|drop] [--reliability-splits 5] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/slopes/slopes.csv, students.csv, summary.json, run.log
"""
import argparse
import json
from pathlib import Path

from qscorer.config import SlopeParams, load_dataset_config
from qscorer.slopes import run_slopes


def main():
    defaults = SlopeParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a name under configs/datasets/ or a path to a config file")
    ap.add_argument("--max-opportunities", default=defaults.max_opportunities, help="an integer, or 'none' for no cap")
    ap.add_argument("--min-support", type=int, default=defaults.min_support)
    ap.add_argument("--timeouts", choices=["keep", "drop"], default=defaults.timeouts)
    ap.add_argument("--reliability-splits", type=int, default=defaults.reliability_splits)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--out", default="results", help="results root; files go to <out>/<dataset>/slopes/")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. 'no-cap' -> slopes-no-cap/")
    args = ap.parse_args()

    cap = None if str(args.max_opportunities).lower() == "none" else int(args.max_opportunities)
    params = SlopeParams(max_opportunities=cap, min_support=args.min_support, timeouts=args.timeouts,
                         reliability_splits=args.reliability_splits, seed=args.seed)
    config = load_dataset_config(args.dataset)
    result = run_slopes(config, params)

    out = Path(args.out) / config.name / ("slopes" + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "slopes.csv", index=False)
    result.students.to_csv(out / "students.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2)
    with open(out / "run.log", "w") as f:
        for k, v in result.log.items():
            f.write(f"{k}: {v}\n")

    s = result.summary
    q, p, sl = s["questions"], s["prior"], s["slopes"]
    print(f"{config.name}: {s['fit']['observations']} observations, {q['supported']} of {q['in_file']} questions with a posterior, sets {q['sets']}, optimal fit {s['fit']['optimal']}")
    print(f"  prior: mu {p['mu']:.3f}, tau {p['tau']:.3f} after {p['em_iterations']} EM iterations; REML Laplace log marginal {p['laplace_log_marginal']:.1f}")
    print(f"  slopes: median {sl.get('median', float('nan')):.3f} (sd {sl.get('median_sd', float('nan')):.3f}), {sl.get('share_positive', float('nan')):.0%} positive, "
          f"P(slope > 0) >= 0.95 for {sl.get('share_p_positive_at_least_0.95', float('nan')):.0%}, <= 0.70 for {sl.get('share_p_positive_at_most_0.70', float('nan')):.0%}")
    if s["jump"]:
        j = s["jump"]
        print(f"  jump (baseline to first practice attempt): {j['questions']} questions, median {j['median']:.3f} logits (sd {j['median_sd']:.3f}), {j['share_positive']:.0%} positive")
    if s["reliability"]:
        m = s["reliability"]["median"]
        print(f"  split-half reliability (Spearman-Brown): slope {m['slope_spearman_brown']:.2f} (r {m['slope_pearson_r']:.2f}, rho {m['slope_spearman_rho']:.2f}), jump {m['jump_spearman_brown']:.2f}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
