"""Predict learning for new questions: score them against the checklist with Gemini or Jev, apply that model's fitted regression, list their flaws.

Run:  uv run python scripts/predict_questions.py --questions my-questions.jsonl --out my-predictions
          [--backend gemini|jev] [--api openrouter|direct] [--model FILE] [--cutoff fixed|relative] [--samples N] [--concurrency 16]
          [--context "..."] [--model-id ID] [--base-url URL]
In:   a JSONL file, one question per line, in the project's format:
      {"id": "q1", "question": {"stem": "...", "choices": [{"label": "a", "text": "..."}, ...]}, "answerKey": "b", "feedback-text": "..."}
      --backend picks the model that scores the checklist: gemini (the default; models/spacing-exp4/model.json) or jev (TypeSafe's Jev;
      models/spacing-exp4-jev/model.json). Each has its own regression, fitted on its own scores; the checklist and the list of flaws are
      the same. --api picks how the model is reached: openrouter (the default, OPENROUTER_API_KEY) or direct (GEMINI_API_KEY for Gemini,
      TYPESAFE_API_KEY for Jev). Gemini is called once per question by default; Jev three times, averaged, as its model was fitted.
Out:  <out>/predictions.csv: id, predicted_slope (logits per log-attempt, the pooled scale of the training data), label ("improvement
      expected", or "low improvement expected" below the cutoff), flaws (the checklist items present that Gemini consistently claimed
      harm improvement; present means half the calls for Gemini, a mean probability of at least 0.7 for Jev). The cutoff is the 25th
      percentile of the training predictions (--cutoff fixed, the default) or of this batch's predictions (--cutoff relative: the bottom
      quarter of the questions in the file; at least 4 questions, and 20 or more for a stable cutoff; adding questions to the file can
      change earlier labels). <out>/scores.csv (the checklist scores), <out>/scores-by-call.csv, <out>/responses.jsonl (raw, resumable),
      <out>/summary.json. Keys in the environment or a .env file.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from qscorer.config import PresenceParams
from qscorer.data import load_questions
from qscorer.jev import model_id as jev_model_id
from qscorer.jev import run_jev
from qscorer.llm import ROOT
from qscorer.predict import apply, label_cutoff
from qscorer.presence import run_presence

MODELS = {"gemini": ROOT / "models" / "spacing-exp4" / "model.json", "jev": ROOT / "models" / "spacing-exp4-jev" / "model.json"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True, help="JSONL of questions to predict")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--backend", choices=["gemini", "jev"], default="gemini", help="the model that scores the checklist: gemini (the default) or jev")
    ap.add_argument("--api", choices=["openrouter", "direct"], default="openrouter",
                    help="openrouter (OPENROUTER_API_KEY, the default) or direct (Google's API with GEMINI_API_KEY, or TypeSafe's with TYPESAFE_API_KEY)")
    ap.add_argument("--model", default=None, help="the fitted predictor (default: the one shipped for --backend)")
    ap.add_argument("--cutoff", choices=["fixed", "relative"], default="fixed",
                    help="fixed: below the training predictions' 25th percentile; relative: the bottom quarter of the questions in this file")
    ap.add_argument("--samples", type=int, default=None,
                    help="calls per question, averaged (default: 1 for gemini, where the score is the share saying present; for jev the 3 its model was fitted with)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--context", default=None, help="the sentence about how the questions are used (default: the training dataset's)")
    ap.add_argument("--model-id", default=None, help="the model id on the API (default: the version the regression was fitted on)")
    ap.add_argument("--base-url", default=None, help="gemini with --api openrouter only: another server of the OpenAI format")
    args = ap.parse_args()

    model_file = Path(args.model) if args.model else MODELS[args.backend]
    model = json.load(open(model_file))
    scoring = model["scoring"]
    engine = scoring.get("engine", "gemini")
    if engine != args.backend:
        raise SystemExit(f"{model_file} was fitted on {engine} scores; use --backend {engine}, or a model fitted on {args.backend} scores")
    samples = args.samples or (scoring.get("samples", 1) if args.backend == "jev" else 1)
    questions = load_questions(Path(args.questions))
    if args.cutoff == "relative":
        if len(questions) < 4:
            raise SystemExit(f"--cutoff relative needs at least 4 questions, the file has {len(questions)}; use --cutoff fixed")
        if len(questions) < 20:
            print(f"warning: --cutoff relative with {len(questions)} questions; the label depends on which few questions are in the file")
    out = Path(args.out)
    context = args.context or scoring["context"]
    if args.backend == "jev":
        result = run_jev(questions, list(questions), model["items"], context, scoring["rewordings"], out, api=args.api,
                         model=args.model_id or jev_model_id(scoring["model"], args.api), samples=samples, concurrency=args.concurrency)
    else:
        backend = "gemini" if args.api == "direct" else "openrouter"
        model_id = args.model_id or (("google/" + scoring["model"]) if backend == "openrouter" and "/" not in scoring["model"] else scoring["model"])
        params = PresenceParams(backend=backend, base_url=args.base_url, model=model_id, thinking=scoring["thinking"], temperature=scoring["temperature"],
                                samples=samples, concurrency=args.concurrency, max_output_tokens=8192)   # a call uses about 4k; OpenRouter checks the budget against the key's credit
        result = run_presence(questions, list(questions), model["items"], context, scoring["system"], params, out)
    result.samples.to_csv(out / "scores-by-call.csv", index=False)
    result.scores.to_csv(out / "scores.csv", index=False)
    predictions = apply(model, result.scores, cutoff=args.cutoff)
    predictions.to_csv(out / "predictions.csv", index=False)
    cut = label_cutoff(model, predictions.predicted_slope.to_numpy(), args.cutoff)
    with open(out / "summary.json", "w") as f:
        json.dump(result.summary | {"engine": args.backend, "model": str(model_file), "trained_on": model["dataset"],
                  "flaw_present": model.get("params", {}).get("flaw_present", 0.5), "cutoff": {"mode": args.cutoff, "value": cut,
                  "quantile": model.get("params", {}).get("label_quantile", 0.25), "questions": len(predictions)}}, f, indent=2)
    s = result.summary
    low = (predictions.label == "low improvement expected").sum()
    print(f"{len(predictions)} of {len(questions)} questions predicted with {args.backend} ({s['answered']} of {s['calls']} calls answered); "
          f"{low} low improvement expected (cutoff {args.cutoff}, {cut:.3f}); {(predictions.flaws != '').sum()} with flaws")
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 100)
    print(predictions.round(3).head(20).to_string(index=False))
    print(f"  wrote {out}/predictions.csv")


if __name__ == "__main__":
    main()
