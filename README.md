# llm-question-scorer

Predicts how much students will improve on a multiple-choice question with practice. A model scores each question against a checklist of 45 observable features of the stem, the answer options and the feedback, and a regression fitted on classroom practice data turns the scores into a predicted learning slope, a ranking among the questions it was trained on, and the features that raised or lowered each question's prediction.

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

`--model` is the fitted predictor shipped in the repository, `--questions` your file, `--out` a folder for the results. One call scores one question, takes about half a minute and costs about a cent; calls run 16 at a time. If the run is interrupted, running the same command again continues where it stopped. Two optional settings: `--samples 2` scores each question twice and averages the scores; `--context "..."` replaces the sentence that tells the model how the questions are used (the default describes weekly practice sessions with feedback after each answer).

## Run the HTTP API

The API scores up to 20 questions in one request and returns the predictions as JSON. Install the extra dependencies and start the server:

```
pip install -e ".[api]"
QSCORER_API_TOKEN=choose-a-long-random-token uvicorn qscorer.api:app --host 0.0.0.0 --port 8000
```

The server reads `OPENROUTER_API_KEY` from the environment or `.env`, as the script does. `QSCORER_MODEL` selects another `model.json`.

Every endpoint except `GET /health` requires the header `Authorization: Bearer <token>`.

- `POST /predict` takes `{"questions": [...], "samples": 1, "context": "..."}`. The questions use the format above. `samples` (1 to 3) and `context` are optional. All calls of a batch run in parallel, so a request takes about 30 seconds.
- `GET /features` returns the 45 features with their coefficients.
- `scorer` in the `/predict` body selects the model that scores the questions: `gemini-flash` (the default, `google/gemini-3.8-flash`) or `jev` (`typesafe/jev-router`). The regression was fitted on Gemini Flash scores. The response field `served_by` counts the models that answered the calls.
- `GET /docs` shows the interactive OpenAPI page.

A question with no usable scoring call goes in `failed`, and the others still get predictions. The server returns 502 when no question can be scored. The JSON Schema files for the request and the responses are in `schemas/`. Run `python scripts/export_api_schemas.py` after changing the shapes in `qscorer/api.py`.

```
curl -s localhost:8000/predict -H "Authorization: Bearer $QSCORER_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"questions": [{"id": "q2", "question": {"stem": "A 2 kg cart is pushed with a net force of 6 N. What is its acceleration?", "choices": [{"label": "a", "text": "3 m/s^2"}, {"label": "b", "text": "12 m/s^2"}]}, "answerKey": "a", "feedback-text": "a = F / m = 3 m/s^2."}]}'
```

## Compare scorers

```
python scripts/compare_scorers.py --questions bank.jsonl --out compare-jev --n 100
```

The script draws 100 questions from the file and scores them with `gemini-flash` and `jev`. Then it compares the two scorers feature by feature (agreement and Cohen's kappa) and in their predictions (Spearman correlation of the predicted slopes, share in the same class, gap in percentile). The results go to `summary.json`, `questions.csv` and `features.csv` in the `--out` folder. A second run sends only the missing calls.

## Interpret results

`my-predictions/predictions.csv` has one row per question:

| column | meaning |
|---|---|
| `id` | the question's id |
| `predicted_slope` | the predicted gain in accuracy with practice, on the scale of the training data (logits of accuracy per log-attempt; the training median was 0.24) |
| `percentile` | where the prediction falls among the questions the model was trained on (0.90 means above 90% of them) |
| `class` | `higher` or `lower` than the training median |
| `raising` | the features present in this question that raised its prediction, largest effect first, with each one's contribution |
| `lowering` | the features present that lowered it |
| `f01` to `f45` | the question's score on each checklist feature: the share of calls that found it present (0 or 1 with one call) |

`my-predictions/features.csv` describes the 45 features: `name`, `definition` and `criterion` (what counts as present), `coefficient` and `direction` (whether its presence predicts more or less learning), `sign_stability` (the share of bootstrap refits in which the direction held; above 0.9 is a reliable direction), `prevalence` (how common it was in the training questions) and `claimed_good_share` (how often the model that proposed the feature called it beneficial). The claimed and the fitted directions often disagree: features that make the correct option easier to pick out, such as a longer correct option or wording shared between stem and key, predict more improvement with practice, while features the model called beneficial, such as scenario-based stems and mechanistic feedback, predict less. The predictions measure improvement on the question, not understanding.

The predictions are a ranking, not a calibrated score. On the data the model was trained on, the ranking correlates with observed learning at about 0.23 (Spearman) within a question bank; on a second dataset the correlation was weaker. Use them to screen and compare questions, not as a verdict on any one of them.
