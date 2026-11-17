"""Review new questions for teachers: flag common item-writing problems with a quote and a suggested fix, and describe what each question does.

Run:  uv run python scripts/review_questions.py --model models/spacing-exp4/model.json --questions my-questions.jsonl --out my-review
          [--review models/spacing-exp4/review.json] [--backend openrouter|gemini] [--base-url URL] [--model-id ID] [--samples 1]
          [--threshold 0.5] [--concurrency 16] [--context "..."] [--title "..."]
In:   a JSONL file of questions in the project's format (see scripts/predict_questions.py).
Out:  <out>/report.html, one card per question, most flags first: the question, each flagged problem with the words that show it, why it
      matters and one way to revise, and grey tags for what the question does. <out>/report.jsonl, the same as data.
      <out>/internal/: the checklist scores (scores.csv, scores-by-call.csv), the check calls (flags.csv), the raw logs (resumable), and
      predictions.csv, the predicted slope, kept for later comparison with student data and not shown to teachers.
How:  each question is scored against the checklist as in scripts/predict_questions.py; the review file sorts the checklist items into
      flaws, features and hidden ones; a flaw scored at or above the threshold is checked by one more call per question, which also
      writes the quote, the explanation and the suggestion.
"""
import argparse
import json
from pathlib import Path

from qscorer.config import PresenceParams
from qscorer.data import load_questions
from qscorer.predict import apply
from qscorer.presence import run_presence
from qscorer.review import build_report, load_groups, render_html, run_review


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.json from scripts/run_predict.py")
    ap.add_argument("--questions", required=True, help="JSONL of questions to review")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--review", default=None, help="the review file sorting the checklist into flaws, features and hidden items (default: review.json next to the model)")
    ap.add_argument("--backend", choices=["gemini", "openrouter"], default="openrouter", help="openrouter (OPENROUTER_API_KEY, the default) or gemini (GEMINI_API_KEY)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model-id", default=None, help="model id on the backend (default: the one the scores were trained with)")
    ap.add_argument("--samples", type=int, default=1, help="scoring calls per question; the score is the share saying present")
    ap.add_argument("--threshold", type=float, default=0.5, help="score at which a flaw is checked and a feature is shown")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--context", default=None, help="the sentence about how the questions are used (default: the training dataset's)")
    ap.add_argument("--title", default="Question review", help="the report's title")
    args = ap.parse_args()

    model = json.load(open(args.model))
    groups = load_groups(Path(args.review) if args.review else Path(args.model).parent / "review.json")
    scoring = model["scoring"]
    model_id = args.model_id or (("google/" + scoring["model"]) if args.backend == "openrouter" and "/" not in scoring["model"] else scoring["model"])
    params = PresenceParams(backend=args.backend, base_url=args.base_url, model=model_id, thinking=scoring["thinking"], temperature=scoring["temperature"],
                            samples=args.samples, concurrency=args.concurrency, max_output_tokens=8192)
    context = args.context or scoring["context"]
    questions = load_questions(Path(args.questions))
    out = Path(args.out); internal = out / "internal"

    result = run_presence(questions, list(questions), model["items"], context, scoring["system"], params, internal)
    result.samples.to_csv(internal / "scores-by-call.csv", index=False)
    result.scores.to_csv(internal / "scores.csv", index=False)
    apply(model, result.scores).to_csv(internal / "predictions.csv", index=False)

    flags = run_review(questions, result.scores, model["items"], groups, context, scoring["system"], params, internal / "review-responses.jsonl", args.threshold)
    flags.to_csv(internal / "flags.csv", index=False)
    report = build_report(questions, result.scores, flags, model["items"], groups, args.threshold)
    with open(out / "report.jsonl", "w") as f:
        for r in report:
            f.write(json.dumps(r) + "\n")
    (out / "report.html").write_text(render_html(report, questions, args.title))

    checked = flags[flags.checked]
    summary = result.summary | {"model": args.model, "trained_on": model["dataset"], "threshold": args.threshold,
                                "review": {"candidates": int(len(flags)), "checked": int(len(checked)), "confirmed": int(checked.applies.sum()),
                                           "quotes_found": int(checked.quote_found.fillna(False).sum()),
                                           "questions_flagged": sum(1 for r in report if r["flaws"])}}
    with open(internal / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    v = summary["review"]
    print(f"{len(report)} of {len(questions)} questions reviewed; {v['questions_flagged']} with flags. "
          f"Flaws: {v['candidates']} candidates, {v['checked']} checked, {v['confirmed']} confirmed ({v['quotes_found']} quotes found in the question).")
    for item, n in flags[flags.applies.fillna(False).astype(bool)].item.value_counts().items():
        print(f"  {item} {groups[item]['label']}: {n}")
    print(f"  wrote {out}/report.html")


if __name__ == "__main__":
    main()
