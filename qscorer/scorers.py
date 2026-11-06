"""The models that can score questions against the checklist, by name, and the parameters of a scoring run.

The fitted regression in `model.json` was trained on Gemini Flash scores. Another scorer feeds its scores into the same
regression, so its predictions are only as good as its agreement with Gemini Flash feature by feature;
`scripts/compare_scorers.py` measures that agreement.

`jev` is OpenRouter's `typesafe/jev-router`. OpenRouter describes it as a router billed at the price of the model it
forwards each request to. The response body names the model that served the call, and `served_by` counts them.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from qscorer.config import PresenceParams

SCORERS = {
    "gemini-flash": "google/gemini-3.8-flash",
    "jev": "typesafe/jev-router",
}
DEFAULT_SCORER = "gemini-flash"


def presence_params(scoring: dict, scorer: str, samples: int = 1, concurrency: int = 16,
                    timeout: float = 120.0, passes: int = 2) -> PresenceParams:
    """Scoring parameters for `scorer`, with the sampling settings the model was trained with."""
    return PresenceParams(backend="openrouter", model=SCORERS[scorer], thinking=scoring["thinking"], temperature=scoring["temperature"],
                          samples=samples, concurrency=concurrency, max_output_tokens=8192, timeout=timeout, passes=passes)


def read_log(log: Path) -> tuple[dict[str, str], Counter, float]:
    """From a `responses.jsonl`: the last error of each question, the count of calls by the model that served
    them, and the total cost OpenRouter reported. Failed calls stay out of `run_presence`'s records, so the log is
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
