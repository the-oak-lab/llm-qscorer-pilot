"""Export one dataset's questions with the observed learning slope and our predictions, for comparing other models against ours.

Run:  uv run python scripts/export_predictions.py --dataset spacing-exp4 [--predict DIR] [--baseline DIR] [--out DIR]
In:   results/<dataset>/predict/predictions.csv (from scripts/run_predict.py), results/<dataset>/slopes/slopes.csv, the dataset's questions,
      and results/<dataset>/baseline/ratings.csv when present
Out:  <out>/questions.csv, one row per question: its text, the Stage 0 slope (the target), our predictions and the Stage 1 ratings;
      <out>/README.md, what each column means and the reference correlations computed from the file
      (default <out> is results/<dataset>/export)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qscorer.config import PredictParams, load_dataset_config
from qscorer.data import load_questions, render_question
from qscorer.predict import labels

COLUMNS = [
    ("id", "question id"),
    ("set", "item set (one class's question bank; each set was seen by different students, so compare predictions within a set)"),
    ("stem", "question stem"),
    ("option_a", "option (a)"), ("option_b", "option (b)"), ("option_c", "option (c)"), ("option_d", "option (d)"),
    ("answer_key", "label of the correct option"),
    ("feedback", "feedback shown after each answer"),
    ("question_text", "stem, options, correct answer and feedback in one block, exactly as our prompts show the question to a model"),
    ("slope", "TARGET: posterior mean of the practice slope, the gain in accuracy with practice (logits of accuracy per log-attempt)"),
    ("slope_sd", "posterior standard deviation of the slope; our ridge weights questions by 1 / slope_sd^2"),
    ("p_positive", "posterior probability that the slope is positive"),
    ("label", "'high' when p_positive >= 0.95, 'low' when <= 0.70, empty otherwise; used for the AUC"),
    ("n_students", "students who practised the question"),
    ("pretest_accuracy", "accuracy on the question at pretest, before practice"),
    ("checklist_cv", "OUR MAIN PREDICTION: the checklist ridge fitted without this question's item set (leave-one-set-out); "
                     "use this one for a fair comparison. It has no intercept, so its level differs between sets"),
    ("checklist_in_sample", "the checklist ridge fitted on all 341 questions, this one included; optimistic, for reference only"),
    ("stage1_improve", "Stage 1 baseline: a model's direct 0-100 rating of how much students will improve with practice (mean of 3 calls)"),
    ("stage1_known", "Stage 1 baseline: 100 minus a model's 0-100 rating of how many students already know the answer (mean of 3 calls)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--predict", default=None, help="the Stage 3.3 folder (default results/<dataset>/predict)")
    ap.add_argument("--baseline", default=None, help="the Stage 1 folder (default results/<dataset>/baseline, skipped if missing)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_dataset_config(args.dataset)
    root = Path("results") / config.name
    pred = pd.read_csv(Path(args.predict or root / "predict") / "predictions.csv")
    slopes = pd.read_csv(root / "slopes" / "slopes.csv")
    questions = load_questions(Path(config.questions))

    rows = []
    for qid in pred.id:
        q = questions[qid]
        options = {c["label"]: c["text"] for c in q["question"]["choices"]}
        rows.append({"id": qid, "stem": q["question"]["stem"], **{f"option_{k}": options.get(k, "") for k in "abcd"},
                     "answer_key": q["answerKey"], "feedback": q.get("feedback-text", "") or "", "question_text": render_question(q)})
    t = pd.DataFrame(rows).merge(slopes[["id", "set", "slope", "slope_sd", "p_positive", "n_students", "pretest_accuracy"]], on="id")
    t["label"] = labels(t, PredictParams()).to_numpy()
    t = t.merge(pred[["id", "cv_predicted_slope", "predicted_slope"]].rename(columns={"cv_predicted_slope": "checklist_cv", "predicted_slope": "checklist_in_sample"}), on="id")
    baseline = Path(args.baseline) if args.baseline else root / "baseline"
    if (baseline / "ratings.csv").exists():
        r = pd.read_csv(baseline / "ratings.csv").pivot(index="id", columns="prompt", values="rating")
        t = t.merge(r.rename(columns=lambda c: f"stage1_{c}").reset_index(), on="id", how="left")
    t = t[[c for c, _ in COLUMNS if c in t.columns]].sort_values(["set", "id"])

    # reference numbers: weighted mean over sets of the within-set Spearman correlation with the slope, and the high-vs-low AUC within sets
    preds = [c for c in ("checklist_cv", "checklist_in_sample", "stage1_improve", "stage1_known") if c in t.columns]
    ref = []
    for c in preds:
        within = [(spearmanr(g[c], g.slope)[0], len(g)) for _, g in t.groupby("set")]
        pairs = [(hi, lo) for _, g in t.groupby("set") for hi in g[c][g.label == "high"] for lo in g[c][g.label == "low"]]
        ref.append({"column": c, "within_set_spearman": sum(r_ * n for r_, n in within) / sum(n for _, n in within),
                    "per_set": ", ".join(f"{r_:.3f}" for r_, _ in within),
                    "auc_high_vs_low_within_sets": float(np.mean([(h > l) + 0.5 * (h == l) for h, l in pairs]))})
    ref = pd.DataFrame(ref)

    out = Path(args.out) if args.out else root / "export"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "questions.csv", index=False)
    sets = ", ".join(f"{s} ({n})" for s, n in t.groupby("set").size().items())
    table = "\n".join(f"| `{c}` | {d} |" for c, d in COLUMNS if c in t.columns)
    reftable = "\n".join(f"| `{r.column}` | {r.within_set_spearman:.3f} | {r.per_set} | {r.auc_high_vs_low_within_sets:.3f} |" for r in ref.itertuples())
    readme = f"""# {config.name}: questions, observed learning and predictions

`questions.csv` has one row per question: {len(t)} questions in {t.set.nunique()} item sets, {sets}.

## Columns

| column | meaning |
|---|---|
{table}

## How to compare a new model

Each item set was practised by a different group of students, so slopes are not comparable across sets. Score a model by the Spearman
correlation of its prediction with `slope` within each set, then average over sets weighted by set size. Our checklist model was chosen
and evaluated this way. Compare against `checklist_cv`, which never saw the set it predicts; `checklist_in_sample` saw every slope and
looks better than it is. If your model is fitted on these slopes, hold out one set at a time in the same way.

Reference numbers from this file:

| prediction | within-set Spearman (weighted) | per set | AUC high vs low, pairs within a set |
|---|---|---|---|
{reftable}

The slope is estimated with noise (see `slope_sd`), so no model can reach a correlation of 1.
"""
    (out / "README.md").write_text(readme)
    print(ref.round(3).to_string(index=False))
    print(f"wrote {out}/questions.csv ({len(t)} rows) and README.md")


if __name__ == "__main__":
    main()
