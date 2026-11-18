# llm-question-scorer

Predicts how much students will improve on a multiple-choice question with practice, and lists the question's flaws. For each question you get three things:

- **a predicted slope**: how much accuracy is expected to rise with practice;
- **a label**: `improvement expected` or `low improvement expected`, from a cutoff on the predicted slope;
- **a list of flaws**: features of the question that Gemini consistently judged harmful to improvement.

A model scores each question against a checklist of 45 observable features of the stem, the answer options and the feedback. The model is Gemini (the default) or TypeSafe's Jev (see [Choosing Gemini or Jev](#choosing-gemini-or-jev)). A regression fitted on classroom practice data turns the 45 scores into the predicted slope. Each model has its own regression, fitted on its own scores. The flaws are the checklist features present in the question that are on a fixed list of 11, the same list for both models (see [Flaws](#flaws)).

## Set up

Python 3.12 or later. Clone the repository and install it:

```
git clone https://github.com/the-oak-lab/llm-qscorer-pilot.git
cd llm-qscorer-pilot
pip install -e .
```

The scoring calls Gemini or Jev through OpenRouter. Create a file named `.env` in the `llm-qscorer-pilot` folder with your OpenRouter API key (from openrouter.ai, under Keys). The file is ignored by git, so the key stays on your machine. One OpenRouter key works for both models:

```
OPENROUTER_API_KEY=sk-or-...
```

To call a model's own API instead (`--api direct`), put `GEMINI_API_KEY` (from Google AI Studio) for Gemini, or `TYPESAFE_API_KEY` (from TypeSafe) for Jev, in the same file.

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

Score every question in a file with Gemini:

```
python scripts/predict_questions.py --questions my-questions.jsonl --out my-predictions
```

Score them with Jev instead:

```
python scripts/predict_questions.py --backend jev --questions my-questions.jsonl --out my-predictions-jev
```

Score a batch and label the bottom quarter of the batch as low:

```
python scripts/predict_questions.py --questions my-questions.jsonl --out my-predictions --cutoff relative
```

The file can hold any number of questions. A single question works with the default cutoff; `--cutoff relative` needs at least 4 (see [Choosing the cutoff](#choosing-the-cutoff)).

One call scores one question, and calls run 16 at a time. A Gemini call takes about half a minute and costs about a cent. A Jev call takes under a second and costs a small fraction of a cent. If a run is interrupted, run the same command again: it reuses the calls already made and continues.

| option | meaning |
|---|---|
| `--questions` | your JSONL file |
| `--out` | a folder for the results; it is created if needed |
| `--backend` | `gemini` (the default) or `jev`: the model that scores the checklist |
| `--api` | `openrouter` (the default, needs `OPENROUTER_API_KEY`) or `direct`: the model's own API, Google's (needs `GEMINI_API_KEY`) or TypeSafe's (needs `TYPESAFE_API_KEY`) |
| `--model` | the fitted predictor; by default the one that ships for the backend, `models/spacing-exp4/model.json` for Gemini and `models/spacing-exp4-jev/model.json` for Jev. A predictor fitted on one model's scores is refused with the other model. |
| `--cutoff` | `fixed` (the default) or `relative`; how the label is decided |
| `--samples` | Gemini only: calls per question (default 1); with 2, each question is scored twice and the scores averaged, which costs twice as much. Jev is asked once. |
| `--context` | the sentence that tells the model how the questions are used; the default describes science classes with a pretest and practice sessions spread over several weeks, with feedback after each answer |
| `--concurrency` | calls run at the same time (default 16) |

## Read the results

`my-predictions/predictions.csv` has one row per question:

| column | meaning |
|---|---|
| `id` | the question's id |
| `predicted_slope` | the predicted gain in accuracy with practice, in logits of accuracy per log-attempt, on the scale of the training data; the predictions for the training questions ranged from 0.00 to 0.48 with Gemini and from −0.06 to 0.46 with Jev, with a median of 0.22 for both |
| `label` | `low improvement expected` when `predicted_slope` is below the cutoff, `improvement expected` otherwise |
| `flaws` | the flaws found in the question, each as id and name, separated by `; `; empty when there are none |

For example (slopes rounded):

```
id,predicted_slope,label,flaws
mcq-34476,0.241,improvement expected,
mcq-36475,0.302,improvement expected,f18 Correct-Option Length and Elaboration Asymmetry; f25 Absolute Qualifiers in Distractor Options
mcq-36539,0.209,improvement expected,f05 Numerical and Structural Cueing Artifacts
```

The folder also holds `summary.json` (the model used, the number of calls, tokens, and the cutoff used), `scores.csv` (each question's score on all 45 checklist features: the share of calls that found it present with Gemini, the probability that it is present with Jev), `scores-by-call.csv`, and `responses.jsonl` (the raw calls, which let a rerun continue).

### Choosing the cutoff

- `--cutoff fixed` (the default): a question is `low improvement expected` when its predicted slope is below the 25th percentile of the predictions for the questions the model was trained on: 0.17 with Gemini, 0.16 with Jev. Every question is judged against the same number, so the label does not depend on what else is in the file. Use it for single questions, or when questions arrive one at a time.
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

The list is the same whichever model scores the question, because it comes from the claims made when the checklist was built, not from the scoring. What differs is how a flaw is found in a question. With Gemini, a flaw is present when at least half the calls say so. With Jev, it is present when Jev's probability that it is present is at least 0.7. The two models agree only moderately on which questions have which flaws. Jev almost never reports f45, so with Jev an empty `flaws` column does not mean the question was checked for factual errors.

The full checklist, with each feature's definition, its weight in the regression and whether it counts as a flaw, is in `models/spacing-exp4/features.csv` (Gemini) and `models/spacing-exp4-jev/features.csv` (Jev).

### Choosing Gemini or Jev

Both models score the same checklist, and each regression was fitted on the same 341 questions (spacing experiment 4, high-school science). Both were also tested on a second dataset that neither was fitted on (spacing experiment 3, middle-school science, 224 questions). The correlations are Spearman correlations of the predicted slope with observed learning, within a question bank:

| | Gemini | Jev |
|---|---|---|
| training data, leaving out one question bank at a time | 0.23 | 0.18 |
| second dataset | 0.12 | 0.23 to 0.27 (two runs) |
| second dataset: share of low-learning questions in the bottom quarter of each bank (14% overall) | 16% | 25% |
| the same questions scored twice: correlation of the two runs' predictions | 0.74 | 0.98 |
| time per question | about 30 seconds | under a second |
| cost per question | about a cent | a small fraction of a cent |

Jev ranks questions a little worse on the data it was fitted on and better on the second dataset. With only two datasets, this suggests Jev carries over to new questions better, but it does not settle it. Jev is also faster, cheaper and more consistent from run to run. Gemini is the better choice when you need factual errors reported (f45).

### What the results can and cannot tell you

- **The label and the flaws come from different sources and can disagree.** The label comes from the regression fitted on student data, and the flaws from Gemini's judgement. Some flaws go with *more* improvement in the data. With a correct option longer than the others, for example, students improve on the question by learning the cue. So a question can be labelled `improvement expected` and still have flaws.
- **`low improvement expected` means less than most questions, not none.** With Gemini every training prediction is above zero; with Jev a few are slightly below.
- **The predictions measure improvement on the same question, not understanding.** The training data tested students again on the questions they practised.
- **Much of the signal is one cue.** A correct option longer than the others (f18) has the largest weight in both regressions. The length of the correct option divided by the average length of the others, measured without any model, correlates with observed learning about as well as either model: 0.26 on the training data and 0.22 on the second dataset.
- **The label is only partly validated on new questions.** On a second dataset, Gemini's predictions correlated with observed learning at 0.12, and its label did not pick out low-learning questions better than chance. Jev's correlated at 0.23 to 0.27, and a quarter of the questions it put in the bottom quarter were low-learning, against 14% overall (see [Choosing Gemini or Jev](#choosing-gemini-or-jev)). Treat the label as a screen to check against student data, not as a verdict on any one question.
