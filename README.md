# llm-question-scorer

Predicts how much students will improve on a multiple-choice question with practice. A model scores each question against a checklist of 45 observable features of the stem, the answer options and the feedback. A regression fitted on classroom practice data turns the scores into a predicted learning slope and a label. The checklist items that Gemini judged harmful and that the question has are listed as its flaws.

## Set up

Python 3.12 or later. Clone the repository and install it:

```
git clone https://github.com/the-oak-lab/llm-qscorer-pilot.git
cd llm-qscorer-pilot
pip install -e .
```

The scoring calls Gemini through OpenRouter. Create a file named `.env` in the `llm-qscorer-pilot` folder with your OpenRouter API key (from openrouter.ai, under Keys):

```
OPENROUTER_API_KEY=sk-or-...
```

## Prepare questions

Put the questions in one JSONL file: one JSON object per line, in the format of [KCluster](https://github.com/weiyumou/KCluster#question-format) with one more field, `feedback-text`, holding the feedback students see after answering. Each object has:

- `id`: a unique id for the question.
- `question.stem`: the stem.
- `question.choices`: the answer options, each with a `label` and a `text`.
- `answerKey`: the label of the correct option.
- `feedback-text`: the feedback. Bullet points can be written on one line separated by ` - `, as in the examples, or with line breaks.

```
{"id": "q1", "question": {"stem": "A book is resting on a table. Which force prevents the book from falling through the table?", "choices": [{"label": "a", "text": "Applied force"}, {"label": "b", "text": "Normal force"}, {"label": "c", "text": "Friction"}, {"label": "d", "text": "Gravity"}]}, "answerKey": "b", "feedback-text": "- The normal force is a contact force that acts perpendicular to a surface, pushing back against an object resting on it. - Gravity pulls the book down, but the table's normal force pushes up with equal magnitude, keeping the book in place. - These two forces are balanced, so the book stays still."}
{"id": "q2", "question": {"stem": "A 2 kg cart is pushed with a net force of 6 N. What is its acceleration?", "choices": [{"label": "a", "text": "3 m/s^2"}, {"label": "b", "text": "12 m/s^2"}, {"label": "c", "text": "4 m/s^2"}, {"label": "d", "text": "0.33 m/s^2"}]}, "answerKey": "a", "feedback-text": "Newton's second law gives a = F / m = 6 N / 2 kg = 3 m/s^2. Multiplying force by mass (12 m/s^2) or dividing mass by force (0.33 m/s^2) are the common slips."}
```

Other fields, such as KCluster's `type`, are allowed and ignored. Leave out questions whose stem refers to a figure, since the model does not see images.

## Run predictions

```
python scripts/predict_questions.py --model models/spacing-exp4/model.json --questions my-questions.jsonl --out my-predictions
```

`--model` is the fitted predictor shipped in the repository, `--questions` your file, `--out` a folder for the results. One call scores one question, takes about half a minute and costs about a cent; calls run 16 at a time. If the run is interrupted, running the same command again continues where it stopped. Three optional settings: `--cutoff` sets how the label is decided (see below); `--samples 2` scores each question twice and averages the scores; `--context "..."` replaces the sentence that tells the model how the questions are used (the default describes weekly practice sessions with feedback after each answer).

## Interpret results

`my-predictions/predictions.csv` has one row per question:

| column | meaning |
|---|---|
| `id` | the question's id |
| `predicted_slope` | the predicted gain in accuracy with practice, on the scale of the training data (logits of accuracy per log-attempt; the training median was 0.22) |
| `label` | `low improvement expected` when the prediction is below the cutoff, `improvement expected` otherwise |
| `flaws` | the checklist items found in the question that Gemini judged harmful to improvement, with id and name, separated by `; `; empty when there are none |

The checklist scores are in `my-predictions/scores.csv`, and the checklist itself, with each item's definition and whether it counts as a flaw, is in `models/spacing-exp4/features.csv`.

The label and the flaws come from different sources and can disagree. The label comes from the regression fitted on student data. The flaws come from Gemini's judgement: an item is a flaw when Gemini consistently called it harmful to improvement. That means at least 90% of the features Gemini proposed for it said so, and those claims came from at least 10 separate batches of questions. Some flaws predict *more* improvement in the data, for example a correct option longer than the others: students improve on the question by learning the cue. So a question can be labelled `improvement expected` and still have flaws. The predictions measure improvement on the question, not understanding.

The cutoff is set with `--cutoff`:

- `--cutoff fixed` (the default): below 0.17, the 25th percentile of the predictions for the questions the model was trained on. Every question gets the same cutoff, so it works on a file with one question.
- `--cutoff relative`: the bottom quarter of the questions in your file. This compares questions the way the model was fitted, within one question bank. It needs at least 4 questions and is more stable with 20 or more. It labels at most a quarter of the file as low, fewer when predictions tie. Adding questions to the file moves the cutoff, so earlier labels can change. Score the questions that will be used together in one file.

The cutoff used is recorded in `my-predictions/summary.json`.

Every training prediction is above zero, so `low improvement expected` means less improvement than most questions, not none. On the data the model was trained on, the predictions correlate with observed learning at about 0.23 (Spearman) within a question bank; on a second dataset the correlation was near zero, and the label did not pick out low-learning questions better than chance. Treat the label as a screen to check against student data, not as a verdict on any one question.
