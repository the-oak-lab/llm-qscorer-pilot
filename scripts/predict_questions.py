"""Predict learning for new questions: score them against the checklist with a model, apply the fitted regression, list their flaws.

Run:  uv run python scripts/predict_questions.py --model models/spacing-exp4/model.json --questions my-questions.jsonl --out my-predictions
          [--backend openrouter|gemini] [--base-url URL] [--model-id google/gemini-3.8-flash] [--samples 1] [--concurrency 16] [--context "..."]
In:   a JSONL file, one question per line, in the project's format:
      {"id": "q1", "question": {"stem": "...", "choices": [{"label": "a", "text": "..."}, ...]}, "answerKey": "b", "feedback-text": "..."}
Out:  <out>/predictions.csv: id, predicted_slope (logits per log-attempt, the pooled scale of the training data), label ("improvement
      expected", or "low improvement expected" below the model's cutoff), flaws (the checklist items present that Gemini claimed harm
      improvement). <out>/scores.csv (the checklist scores), <out>/scores-by-call.csv, <out>/responses.jsonl (raw, resumable),
      <out>/summary.json. Keys: OPENROUTER_API_KEY or GEMINI_API_KEY in the environment or a .env file.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import PresenceParams
from qscorer.data import load_questions
from qscorer.predict import apply
from qscorer.presence import run_presence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.json from scripts/run_predict.py")
    ap.add_argument("--questions", required=True, help="JSONL of questions to predict")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default="openrouter", help="openrouter (OPENROUTER_API_KEY, the default) or gemini (GEMINI_API_KEY)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model-id", default=None, help="model id on the backend (default: the one the scores were trained with, e.g. gemini-3.8-flash; on OpenRouter google/gemini-3.8-flash)")
    ap.add_argument("--samples", type=int, default=1, help="calls per question; the score is the share saying present")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--context", default=None, help="the sentence about how the questions are used (default: the training dataset's)")
    args = ap.parse_args()

    model = json.load(open(args.model))
    scoring = model["scoring"]
    model_id = args.model_id or (("google/" + scoring["model"]) if args.backend == "openrouter" and "/" not in scoring["model"] else scoring["model"])
    params = PresenceParams(backend=args.backend, base_url=args.base_url, model=model_id, thinking=scoring["thinking"], temperature=scoring["temperature"],
                            samples=args.samples, concurrency=args.concurrency, max_output_tokens=8192)   # a call uses about 4k; OpenRouter checks the budget against the key's credit
    questions = load_questions(Path(args.questions))
    out = Path(args.out)
    result = run_presence(questions, list(questions), model["items"], args.context or scoring["context"], scoring["system"], params, out)
    result.samples.to_csv(out / "scores-by-call.csv", index=False)
    result.scores.to_csv(out / "scores.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary | {"model": args.model, "trained_on": model["dataset"]}, f, indent=2)
    predictions = apply(model, result.scores)
    predictions.to_csv(out / "predictions.csv", index=False)
    s = result.summary
    low = (predictions.label == "low improvement expected").sum()
    print(f"{len(predictions)} of {len(questions)} questions predicted ({s['answered']} of {s['calls']} calls answered); "
          f"{low} low improvement expected; {(predictions.flaws != '').sum()} with flaws")
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 100)
    print(predictions.round(3).head(20).to_string(index=False))
    print(f"  wrote {out}/predictions.csv")


if __name__ == "__main__":
    main()
