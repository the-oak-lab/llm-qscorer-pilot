"""Stage 3.3 on one dataset: fit the ridge regression of the practice slope on the checklist scores; optionally test it on another dataset.

Run:  uv run python scripts/run_predict.py --dataset spacing-exp4 [--presence DIR] [--checklist FILE] [--test spacing-exp3 --test-presence DIR]
          [--baseline DIR] [--out results] [--tag NAME]
      The Jev model: --presence results/spacing-exp4/presence-jev --test spacing-exp3 --test-presence results/spacing-exp3/presence-jev --tag jev
In:   the Stage 3.2 scores (Gemini's from scripts/run_presence.py or Jev's from scripts/run_presence_jev.py; the model records which, and Jev's
      flaws count as present at a mean probability of 0.7), the Stage 0 slopes, the checklist, and results/<dataset>/canon/direction-items.csv (from
      scripts/run_direction.py: the direction Gemini claimed for each item and the batches behind it, which decide the flaws for either model)
Out:  results/<dataset>/predict[-tag]/model.json (what scripts/predict_questions.py needs), features.csv (coefficient, direction, sign
      stability, prevalence, the claimed direction and the flaw flag per item), predictions.csv (the training questions with their
      predicted slope, label and flaws), cv.csv (one row per penalty), evaluation.json (cross-validated and in-sample, the Stage 1 baseline
      for comparison when given, and the held-out test when given)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import PredictParams, load_dataset_config
from qscorer.jev import PRESENT
from qscorer.predict import evaluate, feature_table, fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--presence", default=None, help="the Stage 3.2 folder (default results/<dataset>/presence)")
    ap.add_argument("--checklist", default=None, help="checklist.json (default results/<dataset>/canon/checklist.json)")
    ap.add_argument("--baseline", default=None, help="a Stage 1 folder whose evaluation is reported next to the model's (default results/<dataset>/baseline if present)")
    ap.add_argument("--test", default=None, help="a held-out dataset to evaluate the model on")
    ap.add_argument("--test-presence", default=None, help="that dataset's Stage 3.2 folder scored with this checklist (default results/<test>/presence)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path(args.out) / config.name
    presence = Path(args.presence) if args.presence else root / "presence"
    checklist = json.load(open(args.checklist or root / "canon" / "checklist.json"))
    scores = pd.read_csv(presence / "scores.csv")
    slopes = pd.read_csv(root / "slopes" / "slopes.csv")
    p = json.load(open(presence / "summary.json"))["params"]
    engine = p.get("engine", "gemini")
    if engine == "jev":
        scoring = {"engine": "jev", "model": p["model"], "api": p["api"], "samples": p.get("samples", 1), "rewordings": p["rewordings"],
                   "context": config.prompt.context}
        params = PredictParams(flaw_present=PRESENT)
    else:
        scoring = {"engine": "gemini"} | {k: p[k] for k in ("model", "thinking", "temperature", "samples", "backend")} | {"context": config.prompt.context, "system": config.prompt.system}
        params = PredictParams()
    direction = root / "canon" / "direction-items.csv"
    claimed = None
    if direction.exists():
        claimed = pd.read_csv(direction).set_index("item")[["claimed_good_share", "bad_batches"]]
    else:
        print(f"  no {direction}: no claimed direction and no flaws (run scripts/run_direction.py first)")
    result = fit(scores, slopes, checklist, params, scoring, config.name, claimed)
    evaluation = {"training": result.evaluation}
    baseline = Path(args.baseline) if args.baseline else root / "baseline"
    if (baseline / "evaluation.json").exists():
        b = json.load(open(baseline / "evaluation.json"))
        evaluation["stage1_baseline"] = {task: {k: e.get(k) for k in ("spearman", "within_sets_weighted", "auc_high_vs_low")} for task, e in b.items() if isinstance(e, dict)}
    if args.test:
        tconfig = load_dataset_config(args.test)
        tpres = Path(args.test_presence) if args.test_presence else Path(args.out) / tconfig.name / "presence"
        tengine = json.load(open(tpres / "summary.json"))["params"].get("engine", "gemini")
        if tengine != engine:
            raise SystemExit(f"the model is fitted on {engine} scores but {tpres} holds {tengine} scores")
        evaluation["test"] = {"dataset": tconfig.name, "presence": str(tpres)} | evaluate(result.model, pd.read_csv(tpres / "scores.csv"), pd.read_csv(Path(args.out) / tconfig.name / "slopes" / "slopes.csv"), params)
    out = root / ("predict" + (f"-{args.tag}" if args.tag else "")); out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.json", "w") as f:
        json.dump(result.model, f, indent=2)
    coef = feature_table(result.model)
    coef.to_csv(out / "features.csv", index=False)
    result.predictions.to_csv(out / "predictions.csv", index=False)
    result.cv.to_csv(out / "cv.csv", index=False)
    with open(out / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    e = result.evaluation
    print(f"{config.name}: {e['training_questions']} questions, {len(checklist)} items, alpha {e['alpha']}; leave-one-set-out within-set Spearman "
          f"{e['cross_validated']['cv_within_sets_spearman']:.3f}, AUC high vs low {e['cross_validated']['cv_auc_high_vs_low']:.3f}; in-sample Spearman {e['in_sample']['spearman']:.3f}")
    if "stage1_baseline" in evaluation:
        print("  Stage 1 baseline:", json.dumps(evaluation["stage1_baseline"]))
    if "test" in evaluation:
        t = evaluation["test"]
        print(f"  held-out {t['dataset']}: {t['questions']} questions, Spearman {t['spearman']:.3f} {t.get('spearman_ci')}, within sets {t['within_sets_weighted']:.3f}, AUC {t['auc_high_vs_low']}")
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    print(coef[["id", "direction", "coefficient", "sign_stability", "prevalence", "claimed_good_share", "bad_batches", "flaw", "name"]].round(3).to_string(index=False))
    m = result.model
    print(f"  label cutoff {m['label_cutoff']:.3f} (training quantile {params.label_quantile}); flaws (claimed bad share >= {params.flaw_share}, from >= {params.flaw_min_batches} batches): {', '.join(m['flaws'])}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
