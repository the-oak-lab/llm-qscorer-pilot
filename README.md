# llm-question-scorer

Predicts how much students will improve on a multiple-choice question with practice, and lists the question's flaws. For each question you get three things:

- **a predicted slope**: how much accuracy is expected to rise with practice;
- **a label**: `improvement expected` or `low improvement expected`, from a cutoff on the predicted slope;
- **a list of flaws**: features of the question that Gemini consistently judged harmful to improvement.

Gemini scores each question against a checklist of 45 observable features of the stem, the answer options and the feedback. A regression fitted on classroom practice data turns the 45 scores into the predicted slope. The flaws are the checklist features present in the question that are on a fixed list of 11 (see [Flaws](#flaws)).

## Set up

Python 3.12 or later. Clone the repository and install it:

```
git clone https://github.com/the-oak-lab/llm-qscorer-pilot.git
cd llm-qscorer-pilot
pip install -e .
```

The scoring calls Gemini through OpenRouter. Create a file named `.env` in the `llm-qscorer-pilot` folder with your OpenRouter API key (from openrouter.ai, under Keys). The file is ignored by git, so the key stays on your machine:

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

Score every question in a file:

```
python scripts/predict_questions.py --model models/spacing-exp4/model.json --questions my-questions.jsonl --out my-predictions
```

Score a batch and label the bottom quarter of the batch as low:

```
python scripts/predict_questions.py --model models/spacing-exp4/model.json --questions my-questions.jsonl --out my-predictions --cutoff relative
```

The file can hold any number of questions. A single question works with the default cutoff; `--cutoff relative` needs at least 4 (see [Choosing the cutoff](#choosing-the-cutoff)).

One Gemini call scores one question. A call takes about half a minute and costs about a cent, and calls run 16 at a time. If a run is interrupted, run the same command again: it reuses the calls already made and continues.

| option | meaning |
|---|---|
| `--model` | the fitted predictor; use `models/spacing-exp4/model.json`, which ships with the repository |
| `--questions` | your JSONL file |
| `--out` | a folder for the results; it is created if needed |
| `--cutoff` | `fixed` (the default) or `relative`; how the label is decided |
| `--samples` | calls per question (default 1); with 2, each question is scored twice and the scores averaged, which costs twice as much |
| `--context` | the sentence that tells Gemini how the questions are used; the default describes science classes with a pretest and practice sessions spread over several weeks, with feedback after each answer |
| `--backend` | `openrouter` (the default, needs `OPENROUTER_API_KEY`) or `gemini` (Google's API directly, needs `GEMINI_API_KEY`) |
| `--concurrency` | calls run at the same time (default 16) |

## Read the results

`my-predictions/predictions.csv` has one row per question:

| column | meaning |
|---|---|
| `id` | the question's id |
| `predicted_slope` | the predicted gain in accuracy with practice, in logits of accuracy per log-attempt, on the scale of the training data; the training questions ranged from 0.00 to 0.48, with a median of 0.22 |
| `label` | `low improvement expected` when `predicted_slope` is below the cutoff, `improvement expected` otherwise |
| `flaws` | the flaws found in the question, each as id and name, separated by `; `; empty when there are none |

For example (slopes rounded):

```
id,predicted_slope,label,flaws
mcq-34476,0.241,improvement expected,
mcq-36475,0.302,improvement expected,f18 Correct-Option Length and Elaboration Asymmetry; f25 Absolute Qualifiers in Distractor Options
mcq-36539,0.209,improvement expected,f05 Numerical and Structural Cueing Artifacts
```

The folder also holds `summary.json` (the number of calls, tokens, and the cutoff used), `scores.csv` (each question's score on all 45 checklist features), `scores-by-call.csv`, and `responses.jsonl` (the raw calls, which let a rerun continue).

### Choosing the cutoff

- `--cutoff fixed` (the default): a question is `low improvement expected` when its predicted slope is below 0.17, the 25th percentile of the predictions for the questions the model was trained on. Every question is judged against the same number, so the label does not depend on what else is in the file. Use it for single questions, or when questions arrive one at a time.
- `--cutoff relative`: the bottom quarter of the questions in your file are `low improvement expected`. This compares questions the way the model was fitted, within one question bank, and it does not rely on the new bank being like the training bank. It needs at least 4 questions and is more stable with 20 or more. It labels at most a quarter of the file as low, fewer when predictions tie. Adding questions to the file moves the cutoff and can change earlier labels, so put the questions that will be used together in one file.

`summary.json` records which cutoff was used and its value.

### Flaws

A flaw is a checklist feature that Gemini consistently judged harmful to improvement. When the checklist was built, Gemini compared questions on which students improved more and less, and proposed features in separate calls for what helps and what hurts. A feature counts as a flaw when at least 90% of Gemini's claims about it said it hurts, and those claims came from at least 10 separate batches of questions. Flaws are not only writing problems; they are whatever Gemini consistently called harmful. There are 11:

| id | part | name | what it means |
|---|---|---|---|
| f05 | options | Numerical and Structural Cueing Artifacts | numbers, units or the layout of the options let students answer by a shortcut |
| f08 | stem | Arbitrary Historical and Conventional Fact Retrieval | the question asks for a name, date or naming convention that cannot be reasoned out |
| f14 | stem | Everyday Intuitive Solvability | the answer follows from everyday common sense, without the science concept |
| f18 | options | Correct-Option Length and Elaboration Asymmetry | the correct option is longer or more detailed than the others |
| f24 | options | Superficially Eliminable or Premise-Conflicting Distractors | a wrong option can be ruled out without knowing the content |
| f25 | options | Absolute Qualifiers in Distractor Options | only wrong options use words such as "all", "only" or "never" |
| f27 | stem | Stem-to-Option Lexical or Informational Cueing | words in the stem point to the correct option |
| f41 | feedback | Restatement Feedback Redundancy | the feedback mostly repeats the question or the answer |
| f42 | feedback | Internal Item-Feedback Discrepancy | the feedback disagrees with the question or the options |
| f43 | feedback | Extraneous Conceptual Content in Feedback | the feedback brings in ideas the question does not need |
| f45 | feedback | Factual or Conceptual Inaccuracy in Item or Feedback | the question or the feedback contains a factual or scientific error |

The full checklist, with each feature's definition and whether it counts as a flaw, is in `models/spacing-exp4/features.csv`.

### What the results can and cannot tell you

- **The label and the flaws come from different sources and can disagree.** The label comes from the regression fitted on student data, and the flaws from Gemini's judgement. Some flaws go with *more* improvement in the data. With a correct option longer than the others, for example, students improve on the question by learning the cue. So a question can be labelled `improvement expected` and still have flaws.
- **`low improvement expected` means less than most questions, not none.** Every training prediction is above zero.
- **The predictions measure improvement on the same question, not understanding.** The training data tested students again on the questions they practised.
- **The label is not yet validated on new questions.** On the data the model was trained on, the predictions correlate with observed learning at about 0.23 (Spearman) within a question bank. On a second dataset the correlation was near zero, and the label did not pick out low-learning questions better than chance. Treat the label as a screen to check against student data, not as a verdict on any one question.
