# llm-question-scorer

LLM-based discovery of question features predictive of learning: Stage 0 measures each question's practice slope from student data, Stage 1 asks a model to predict learning directly, Stage 2 discovers features by contrastive prompting and consolidates them into a checklist of observables, Stage 3 scores questions against the checklist and fits a regression that predicts learning from the scores. The design is in `docs/design.md`, the running log in `docs/logs.md`; `qscorer/` is the package, `scripts/run_*.py` the stage runs.

## Install

```
uv sync                      # in this repository (Python 3.12 or later)
pip install git+<this repository's URL>   # as a package elsewhere
```

## Model backends

Every stage that calls a model takes `--backend`:

- `gemini` (default): the Gemini API through google-genai; key `GEMINI_API_KEY`; model ids such as `gemini-3.8-flash` and `gemini-embedding-2`.
- `openrouter`: the OpenAI chat-completions format over HTTP; key `OPENROUTER_API_KEY`; model ids as OpenRouter names them, such as `google/gemini-3.8-flash` and `google/gemini-embedding-2`. `--base-url https://host/v1` points it at any other server of that format, with the key `OPENAI_API_KEY`.

Keys are read from the environment or from a `.env` file (`NAME=value` lines) in the working directory or the repository. Every raw response is appended to `responses.jsonl` in the run's folder, so an interrupted run resumes and the thought summaries and token counts of every call are kept; a run's `summary.json` records which backend and model answered.

```
uv run python scripts/run_baseline.py --dataset spacing-exp4 --limit 8 --tag pilot
uv run python scripts/run_baseline.py --dataset spacing-exp4 --backend openrouter --model google/gemini-3.8-flash --limit 8 --tag pilot
```

## Predicting learning for new questions

The deliverable for collaborators is one file, `results/<dataset>/predict/model.json`: the checklist (item names, definitions and scoring criteria), the regression coefficients, and the scoring settings. `scripts/predict_questions.py` scores new questions against the checklist with a model and applies the regression:

```
uv run python scripts/predict_questions.py --model results/spacing-exp4/predict/model.json --questions my-questions.jsonl --out my-predictions
uv run python scripts/predict_questions.py --model results/spacing-exp4/predict/model.json --questions my-questions.jsonl --out my-predictions --backend openrouter
```

`my-questions.jsonl` has one question per line in the project's format (`id`, `question.stem`, `question.choices` with `label` and `text`, `answerKey`, `feedback-text`). `my-predictions/predictions.csv` gives each question a predicted practice slope on the training data's scale (logits per log-attempt), its percentile among the training questions, a class (`higher` or `lower` than the training median) and its score on every checklist item. One call per question by default; `--samples 2` scores each question twice and averages. The raw responses are kept in the output folder, so an interrupted run resumes.

How the model is built, stage by stage (each script's docstring has the options):

```
uv run python scripts/run_slopes.py --dataset spacing-exp4        # Stage 0: practice slopes from the student data
uv run python scripts/run_pairs.py --dataset spacing-exp4         # Stage 0.2: high- and low-learning pairs
uv run python scripts/run_batches.py --dataset spacing-exp4       # Stage 2.1: batches of questions for discovery
uv run python scripts/run_discovery.py --dataset spacing-exp4     # Stage 2.2: feature discovery, one batch and condition per call
uv run python scripts/run_canon.py --dataset spacing-exp4         # Stage 2.3: embed, cluster, name, split, consolidate into the checklist
uv run python scripts/run_presence.py --dataset spacing-exp4      # Stage 3.2: score every question against the checklist
uv run python scripts/run_presence.py --dataset spacing-exp3 --checklist results/spacing-exp4/canon/checklist.json   # the held-out dataset, same checklist
uv run python scripts/run_predict.py --dataset spacing-exp4 --test spacing-exp3   # Stage 3.3: fit the ridge regression, test it on exp3
```
