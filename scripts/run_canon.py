"""Stage 2.3 normalisation on one dataset: embed the raw features, cut the Ward tree, name the clusters, split the incoherent ones,
consolidate the leaves into a checklist.

Run:  uv run python scripts/run_canon.py --dataset spacing-exp4 [--discovery DIR] [--k 50] [--min-split 20] [--members-shown 40] [--concurrency 64]
          [--backend gemini|openrouter] [--base-url URL] [--model gemini-3.8-flash] [--embedding-model gemini-embedding-2] [--seed 0] [--out results] [--tag NAME]
Out:  results/<dataset>/canon[-tag]/embeddings.npy, embedded.csv, linkage.npy, features.csv (text, cluster, leaf, item), leaves.csv,
      checklist.json, checklist.csv, mapping.csv (raw feature -> item id), responses.jsonl (naming and consolidation calls), summary.json
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
    ap.add_argument("--k", type=int, default=d.k, help="clusters cut from the Ward tree before naming")
    ap.add_argument("--min-split", type=int, default=d.min_split)
    ap.add_argument("--members-shown", type=int, default=d.members_shown)
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default=d.backend, help="gemini (GEMINI_API_KEY) or openrouter (OPENROUTER_API_KEY)")
    ap.add_argument("--base-url", default=d.base_url, help="openrouter only: another server of the OpenAI format")
    ap.add_argument("--model", default=d.model, help="model id on the backend, for the naming and consolidation calls")
    ap.add_argument("--embedding-model", default=d.embedding_model, help="embedding model id on the backend")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    discovery = Path(args.discovery) if args.discovery else root / "discovery"
    features = pd.read_csv(discovery / "features.csv")
    params = CanonParams(backend=args.backend, base_url=args.base_url, model=args.model, embedding_model=args.embedding_model, k=args.k,
                         min_split=args.min_split, members_shown=args.members_shown, concurrency=args.concurrency, seed=args.seed)
    out = root / ("canon" + (f"-{args.tag}" if args.tag else ""))
    result = run_canon(config, params, features, out)

    result.features.to_csv(out / "features.csv", index=False)
    result.leaves.to_csv(out / "leaves.csv", index=False)
    result.mapping.to_csv(out / "mapping.csv", index=False)
    with open(out / "checklist.json", "w") as f:
        json.dump(result.checklist, f, indent=2)
    pd.DataFrame(result.checklist).assign(sources=lambda t: t.sources.str.join(" ")).to_csv(out / "checklist.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2)

    s = result.summary
    print(f"{config.name}: {s['raw_features']} raw features; {s['k']} clusters, {s['clusters']['incoherent']} incoherent, {s['clusters']['split']} split; "
          f"{s['leaves']['n']} leaves; checklist of {s['checklist']['n']} items ({s['checklist']['merged_items']} merging several leaves), by part {s['checklist']['by_part']}")
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(pd.DataFrame(result.checklist)[["id", "part", "size", "name"]].to_string(index=False))
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
