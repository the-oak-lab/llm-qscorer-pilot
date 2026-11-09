"""The models that can score questions against the checklist, by name, and one entry point that runs either.

- `gemini-flash`: an LLM call per question through OpenRouter's chat completions (`qscorer.presence`), the scorer the
  regression in `model.json` was trained on. The score is the share of calls that found a feature present.
- `haiku`: Claude Haiku 5.5, the same LLM call and prompt as gemini-flash, as a reference for how much the scoring
  model matters. OpenRouter accepts the seed, but Claude has no seed parameter, so a repeat run differs by sampling alone.
- `jev`: TypeSafe's Jev, a System One model, through OpenRouter's decisions endpoint (`qscorer.jev`). The score is Jev's
  probability that the feature is present.

Another scorer's scores go into the same regression, so its predictions are only as good as its agreement with Gemini
Flash feature by feature. `scripts/compare_scorers.py` measures that agreement.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from qscorer.config import PresenceParams
from qscorer.jev import DEFAULT_VARIANT, JEV_MODEL, run_jev
from qscorer.presence import PresenceResult, run_presence

SCORERS = {
    "gemini-flash": "google/gemini-3.8-flash",
    "haiku": "anthropic/claude-haiku-5.5",
    "jev": JEV_MODEL,
}
DEFAULT_SCORER = "gemini-flash"


def presence_params(scoring: dict, samples: int = 1, concurrency: int = 16, timeout: float = 120.0, passes: int = 2,
                    call_seed: int = 0, scorer: str = "gemini-flash") -> PresenceParams:
    """LLM scoring parameters for `scorer` (gemini-flash or haiku), with the sampling settings and output budget the model was trained with. A smaller
    budget truncates the answer on questions that need long reasoning (seen at 8192: ~7.9k reasoning tokens, answer cut at f14)."""
    return PresenceParams(backend="openrouter", model=SCORERS[scorer], thinking=scoring["thinking"], temperature=scoring["temperature"],
                          samples=samples, concurrency=concurrency, max_output_tokens=16384, timeout=timeout, passes=passes,
                          call_seed=call_seed)


def score(scorer: str, questions: dict[str, dict], ids: list[str], model: dict, context: str | None, out: Path,
          samples: int = 1, concurrency: int = 16, timeout: float | None = None, passes: int = 2,
          jev_variant: str = DEFAULT_VARIANT, call_seed: int = 0, engine_factory=None, progress=print) -> PresenceResult:
    """Score `ids` with `scorer` against the model's checklist; raw log in `out/responses.jsonl`, resumed if present.
    `samples` and `call_seed` apply to the LLM scorers only: Jev takes no seed, so it is asked once. `engine_factory(scorer, model_id)`
    replaces the network backend (tests)."""
    scoring = model["scoring"]
    context = context or scoring["context"]
    engine = engine_factory(scorer, SCORERS[scorer]) if engine_factory else None
    if scorer == "jev":
        return run_jev(questions, ids, model["items"], context, out, variant=jev_variant, concurrency=concurrency, timeout=timeout or 60.0,
                       passes=passes, engine=engine, progress=progress)
    params = presence_params(scoring, samples=samples, concurrency=concurrency, timeout=timeout or 120.0, passes=passes, call_seed=call_seed,
                             scorer=scorer)
    return run_presence(questions, ids, model["items"], context, scoring["system"], params, out, engine=engine, progress=progress)


def read_log(log: Path) -> tuple[dict[str, str], Counter, float]:
    """From a `responses.jsonl`: the last error of each question, the count of calls by the model that served
    them, and the total cost OpenRouter reported. Failed calls stay out of the scorers' records, so the log is
    the only place their errors are."""
    errors: dict[str, str] = {}
    served: Counter = Counter()
    cost = 0.0
    if not log.exists():
        return errors, served, cost
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            errors[rec["id"].split("|")[0]] = str(rec["error"])
            continue
        if rec.get("model"):
            served[rec["model"]] += 1
        cost += float((rec.get("usage") or {}).get("cost") or 0.0)
    return errors, served, cost
