"""Stage 2.2 on one dataset: feature discovery on the planned batches, one batch and condition per call.

Run:  uv run python scripts/run_discovery.py --dataset spacing-exp4 [--limit 2] [--samples 2] [--backend gemini|openrouter] [--base-url URL] [--model gemini-3.8-flash]
          [--temperature 1.0] [--thinking medium] [--no-thoughts] [--concurrency 8] [--passes 3] [--seed 0] [--out results] [--tag NAME] [--dry-run]
Out:  results/<dataset>/discovery[-tag]/responses.jsonl (raw, resumable), calls.csv, features.csv, attributions.csv, summary.json,
      prompt-example.txt
A pilot: --limit 2 --tag pilot asks the first two batches of each kind in round 1 under all their conditions.
--dry-run prints a prompt and the call count without calling.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import DiscoveryParams, load_dataset_config
from qscorer.data import load_questions
from qscorer.discovery import build_jobs, run_discovery, select_batches


def main():
    d = DiscoveryParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=None, help="ask only this many batches of each kind, from the first round (pilots)")
    ap.add_argument("--samples", type=int, default=d.samples)
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default=d.backend, help="gemini (GEMINI_API_KEY) or openrouter (OPENROUTER_API_KEY)")
    ap.add_argument("--base-url", default=d.base_url, help="openrouter only: another server of the OpenAI format")
    ap.add_argument("--model", default=d.model, help="model id on the backend")
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], default=d.thinking)
    ap.add_argument("--no-thoughts", action="store_true")
    ap.add_argument("--concurrency", type=int, default=d.concurrency)
    ap.add_argument("--passes", type=int, default=d.passes)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    params = DiscoveryParams(backend=args.backend, base_url=args.base_url, model=args.model, samples=args.samples, temperature=args.temperature, thinking=args.thinking, thoughts=not args.no_thoughts,
                             concurrency=args.concurrency, passes=args.passes, limit=args.limit, seed=args.seed)
    root = Path(args.out) / config.name
    plan = pd.read_csv(root / "batches" / "batches.csv")
    out = root / ("discovery" + (f"-{args.tag}" if args.tag else ""))

    if args.dry_run:
        chosen = select_batches(plan, params)
        jobs = build_jobs(chosen, load_questions(config.questions), config, params)
        print(f"{config.name}: {chosen.batch.nunique()} batches, {len(jobs)} calls to {params.model} on {params.backend} ({params.samples} per batch and condition); would write {out}/")
        shown = next(j for j in jobs if j.id.split("|")[1] == "E7")
        print(f"\n--- {shown.id} ---\n{shown.prompt}\n--- settings ---\n{json.dumps(shown.settings())}")
        return

    result = run_discovery(config, params, plan, out)
    result.calls.to_csv(out / "calls.csv", index=False)
    result.features.to_csv(out / "features.csv", index=False)
    result.attributions.to_csv(out / "attributions.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2)
    (out / "prompt-example.txt").write_text(result.example)

    s = result.summary
    print(f"{config.name}: {s['batches_asked']} batches, {s['calls']} calls, {s['answered']} answered, {s['failed']} failed {s['errors'] or ''}")
    if s["tokens_per_call"]:
        print("  tokens per call: " + ", ".join(f"{k.replace('_tokens', '')} {v:.0f}" for k, v in s["tokens_per_call"].items()) + f"; total {sum(s['tokens'].values()):,}")
    print(f"  features per call: {s['features_per_call']}; evidence quotes found in the question text: {s['evidence_found']:.0%}" if s["evidence_found"] is not None else "")
    for c, e in s["per_condition"].items():
        line = (f"  {c}: {e['calls']} calls, {e['features_per_call']:.1f} features per call, {e['questions_per_feature'] or 0:.1f} questions per feature"
                f" ({e['contrasts_per_feature'] or 0:.1f} contrasts), evidence found {e['evidence_found'] or 0:.0%}")
        if "attributions_to_high_members" in e:
            line += f", attributed to high members {e['attributions_to_high_members']:.0%}"
        if "attributions_to_HIGH_labelled" in e:
            line += f", to HIGH-labelled {e['attributions_to_HIGH_labelled']:.0%}"
        print(line)
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
