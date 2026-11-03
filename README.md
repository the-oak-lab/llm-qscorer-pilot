# llm-question-scorer

LLM-based discovery of question features predictive of learning: Stage 0 measures each question's practice slope from student data, Stage 1 asks a model to predict learning directly, Stage 2 discovers features by contrastive prompting and normalises them into a codebook, Stage 3 scores questions against the codebook. The design is in `docs/design.md`, the running log in `docs/logs.md`; `qscorer/` is the package, `scripts/run_*.py` the stage runs.

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
