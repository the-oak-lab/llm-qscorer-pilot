"""Stage 2.3 normalisation on one dataset: embed the raw features, cluster them, pick the cuts, classify polarity, name the clusters.

Run:  uv run python scripts/run_canon.py --dataset spacing-exp4 [--discovery DIR] [--k-max 400] [--members-shown 40] [--concurrency 64]
          [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/canon[-tag]/embeddings.npy, embedded.csv, linkage.npy, curves.csv, features.csv (text, polarity, cluster at each
      named cut), clusters-k<K>.csv and mapping-k<K>.csv per named cut, responses.jsonl (polarity and naming calls), summary.json
Every step is cached in the folder, so a rerun does no call twice.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.canon import run_canon
from qscorer.config import CanonParams, load_dataset_config


def main():
    d = CanonParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--discovery", default=None, help="the Stage 2.2 folder (default results/<dataset>/discovery)")
    ap.add_argument("--k-max", type=int, default=d.k_max)
    ap.add_argument("--members-shown", type=int, default=d.members_shown)
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    discovery = Path(args.discovery) if args.discovery else root / "discovery"
    features = pd.read_csv(discovery / "features.csv")
    params = CanonParams(k_max=args.k_max, members_shown=args.members_shown, concurrency=args.concurrency, seed=args.seed)
    out = root / ("canon" + (f"-{args.tag}" if args.tag else ""))
    result = run_canon(config, params, features, out)

    result.features.to_csv(out / "features.csv", index=False)
    result.curves.to_csv(out / "curves.csv", index=False)
    for k, t in result.clusters.items():
        t.to_csv(out / f"clusters-k{k}.csv", index=False)
        result.mappings[k].to_csv(out / f"mapping-k{k}.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2)

    s = result.summary
    print(f"{config.name}: {s['raw_features']} raw features embedded in {s['dimensions']} dimensions; elbow at k = {s['cuts']['elbow']}, "
          f"stability-optimal cut at k = {s['cuts']['stability']}; polarity {s['polarity']}")
    stab = result.curves.dropna(subset=["stability"])
    print("  two-call stability by k (observed - baseline): " + ", ".join(f"{int(r.k)}: {r.stability:.3f} ({r.stability_observed:.2f} - {r.stability_baseline:.2f})" for r in stab.itertuples()))
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
    for k, t in result.clusters.items():
        c = s["clusters"][str(k)]
        print(f"\n  cut k = {k}: {c['n']} clusters, {c['named']} named, {c['coherent']} coherent, largest {c['largest']}, singletons {c['singletons']}; the ten largest:")
        print(t.sort_values("size", ascending=False).head(10)[["cluster", "size", "absence_share", "coherent", "name"]].to_string(index=False))
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
