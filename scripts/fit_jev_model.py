"""Fit the slope regression on Jev's scores of the researchers' spacing-exp4 questions and write models/jev-exp4.

Run from the repo root:  python scripts/fit_jev_model.py [--csv datasets/nsfv4/researchers/questions-exp4.csv]
Steps:
  1. Convert the researchers' CSV (341 questions, original option order) to JSONL.
  2. Score every question with one Jev request (`tuned` variant). Resumable: responses go to results/nsfv4/jev-tuned-original-order.
  3. Fit the repo's ridge regression (qscorer.predict.fit) on Jev's raw probabilities (the `.joined` columns) against the
     researchers' slopes. The penalty comes from leave-one-set-out CV over the 4 item sets, as for the Gemini model.
Out: models/jev-exp4/model.json and features.csv. The Gemini calibration is not applied (it lowered held-out accuracy).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qscorer.config import PredictParams
from qscorer.data import load_questions
from qscorer.jev import JEV_MODEL
from qscorer.predict import fit
from qscorer.scorers import read_log, score


def to_jsonl(csv: Path, out: Path) -> None:
    r = pd.read_csv(csv, keep_default_na=False)
    rows = [{"id": a.id, "question": {"stem": a.stem.strip(), "choices": [{"label": c, "text": str(a[f"option_{c}"]).strip()} for c in "abcd"]},
             "answerKey": a.answer_key.strip(), "feedback-text": str(a.feedback).strip(), "type": "mcq"} for _, a in r.iterrows()]
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="datasets/nsfv4/researchers/questions-exp4.csv")
    ap.add_argument("--gemini-model", default="models/spacing-exp4/model.json", help="source of the checklist and the context")
    ap.add_argument("--scores", default="results/nsfv4/jev-tuned-original-order")
    ap.add_argument("--out", default="models/jev-exp4")
    args = ap.parse_args()

    gem = json.loads(Path(args.gemini_model).read_text())
    items = gem["items"]; ids_ = [i["id"] for i in items]
    csv = Path(args.csv); jsonl = csv.with_suffix(".jsonl")
    to_jsonl(csv, jsonl)
    Q = load_questions(jsonl)

    sdir = Path(args.scores); sdir.mkdir(parents=True, exist_ok=True)
    r = score("jev", Q, list(Q), gem, None, sdir, jev_variant="tuned", concurrency=16, timeout=60.0, passes=3)
    r.scores.to_csv(sdir / "scores.csv", index=False)
    errors, _, cost = read_log(sdir / "responses.jsonl")
    print(f"{r.summary['answered']} of {r.summary['calls']} answered; cost ${cost:.4f}; errors {len(errors)}")

    R = pd.read_csv(csv)
    raw = r.scores.set_index("id")[[f"{f}.joined" for f in ids_]].set_axis(ids_, axis=1).reset_index()
    scoring = {"scorer": "jev", "model": JEV_MODEL, "variant": "tuned", "columns": "raw .joined probabilities", "context": gem["scoring"]["context"]}
    f = fit(raw, R[["id", "set", "slope", "slope_sd", "p_positive"]], items, PredictParams(), scoring,
            "spacing-exp4 (researchers' 341, original option order)", boot=1000)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(f.model, indent=2))
    pd.DataFrame(f.model["features"]).to_csv(out / "features.csv", index=False)
    cv = {k: round(v, 3) for k, v in f.evaluation["cross_validated"].items() if k != "alpha"}
    tp = np.percentile(f.model["training_predictions"], [10, 20, 50]).round(3)
    print(f"wrote {out}; alpha {f.model['alpha']}; CV {cv}; training prediction percentiles 10/20/50 {tp}")


if __name__ == "__main__":
    main()
