"""Stage 3.2 (`presence`): score questions against the checklist.

Each question is shown alone, exactly as in Stage 1 (the dataset's context sentence, then the
stem, four options, correct answer and feedback), followed by the checklist and the task: say
for every item whether it is present in this question. One call covers the whole checklist;
the answer is JSON with one boolean per item. A question is asked `samples` times with
different seeds, and its **score** on an item is the share of its calls that said present,
so with two or more samples the per-item agreement of the calls is measured.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qscorer.config import PresenceParams
from qscorer.data import render_question
from qscorer.llm import Job, make_backend, parse

TASK = ("Your task:\nBelow is a checklist of features that a question or its feedback may have. For each feature on the checklist, decide "
        "whether it is present in this question, judging only what is observable in the stem, the options, the correct answer and the feedback, "
        "by the feature's definition and criterion.")
FORMAT = ('Respond in JSON with one field, "features": a list with one object per checklist feature, in the order given, each with "id" '
          '(the feature\'s id) and "present" (true or false).')
SCHEMA = {"type": "object", "required": ["features"], "properties": {"features": {"type": "array", "items": {
    "type": "object", "required": ["id", "present"], "properties": {"id": {"type": "string"}, "present": {"type": "boolean"}}}}}}


@dataclass
class PresenceResult:
    samples: pd.DataFrame     # one row per call: id, sample, one column per item (True/False), tokens, error
    scores: pd.DataFrame      # one row per question: n_ok, one column per item (share of calls saying present)
    summary: dict
    example: str


def render_checklist(checklist: list[dict]) -> str:
    lines = []
    for it in checklist:
        crit = it["criterion"].strip().rstrip(".")
        crit = crit if crit.lower().startswith("count") else f"Count as present when {crit}"
        lines.append(f"{it['id']}. {it['name']}: {it['definition'].strip()} {crit}.")
    return "\n".join(lines)


def build_prompt(q: dict, checklist: list[dict], context: str) -> str:
    head = "\n\n".join([context, "Below is one question with its stem, four answer options, correct answer, and feedback.", render_question(q)])
    return head + "\n\n\n" + TASK + "\n\nChecklist:\n" + render_checklist(checklist) + "\n\n" + FORMAT


def build_jobs(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, system: str, params: PresenceParams) -> list[Job]:
    return [Job(f"{qid}|{k}", build_prompt(questions[qid], checklist, context), system=system, schema=SCHEMA, temperature=params.temperature,
                seed=k, max_output_tokens=params.max_output_tokens, thinking=params.thinking, thoughts=params.thoughts)
            for qid in ids for k in range(params.samples)]


def parse_presence(text: str | None, item_ids: list[str]) -> dict[str, bool] | None:
    """item id -> present, when every item is answered; else None."""
    if not text:
        return None
    try:
        items = json.loads(text).get("features")
    except (ValueError, AttributeError):
        return None
    if not isinstance(items, list):
        return None
    out = {}
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("present"), bool) and str(it.get("id")).strip() in item_ids:
            out.setdefault(str(it["id"]).strip(), it["present"])
    return out if all(i in out for i in item_ids) else None


def make_accept(item_ids: list[str]):
    def accept(raw: dict) -> bool:
        r = parse(raw)
        return r.ok and parse_presence(r.text, item_ids) is not None
    return accept


def sample_table(records: dict[str, dict], jobs: list[Job], item_ids: list[str]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        qid, k = job.id.split("|")
        raw = records.get(job.id)
        r = parse(raw) if raw else None
        answers = parse_presence(r.text, item_ids) if r else None
        rows.append({"id": qid, "sample": int(k), "ok": answers is not None, **{i: (answers[i] if answers else None) for i in item_ids},
                     "thought": r.thought if r else None, "prompt_tokens": r.prompt_tokens if r else None, "output_tokens": r.output_tokens if r else None,
                     "thought_tokens": r.thought_tokens if r else None, "error": (r.error if r else "not asked")})
    return pd.DataFrame(rows)


def score_table(samples: pd.DataFrame, item_ids: list[str]) -> pd.DataFrame:
    ok = samples[samples.ok]
    g = ok.groupby("id")
    t = pd.DataFrame({"n_ok": g.size()})
    for i in item_ids:
        t[i] = g[i].apply(lambda s: float(np.mean([bool(v) for v in s])))
    return t.reset_index()


def agreement(samples: pd.DataFrame, item_ids: list[str]) -> dict:
    """Per item: the share of questions scored present (over all calls), and, for questions with
    at least two calls, the agreement of the first two calls and Cohen's kappa."""
    ok = samples[samples.ok]
    out = {}
    first = ok[ok["sample"] == 0].set_index("id"); second = ok[ok["sample"] == 1].set_index("id")
    both = first.index.intersection(second.index)
    for i in item_ids:
        e = {"prevalence": float(ok[i].astype(bool).mean()) if len(ok) else float("nan")}
        if len(both) >= 5:
            a, b = first.loc[both, i].astype(bool).to_numpy(), second.loc[both, i].astype(bool).to_numpy()
            po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
            e |= {"questions_with_two_calls": int(len(both)), "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}
        out[i] = e
    return out


def select_questions(questions: dict[str, dict], exclude: list[str], limit: int | None, sets: pd.Series | None, seed: int) -> list[str]:
    """The ids scored: the bank minus `exclude`; with `limit`, that many spread over the item sets in turn."""
    ids = [q for q in questions if q not in set(exclude)]
    if limit is None or limit >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    by_set: dict[str, list[str]] = {}
    for q in ids:
        by_set.setdefault(sets.get(q, "set-1") if sets is not None else "set-1", []).append(q)
    queues = [list(rng.permutation(v)) for _, v in sorted(by_set.items())]
    out: list[str] = []
    while len(out) < limit:
        for queue in queues:
            if queue and len(out) < limit:
                out.append(queue.pop())
    return out


def run_presence(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, system: str, params: PresenceParams, out: Path,
                 engine=None, progress=print) -> PresenceResult:
    """Score `ids` against the checklist; raw log in `out/responses.jsonl`, resumed if present."""
    item_ids = [it["id"] for it in checklist]
    jobs = build_jobs(questions, ids, checklist, context, system, params)
    out.mkdir(parents=True, exist_ok=True)
    engine = engine or make_backend(params)
    records = engine.many(jobs, out / "responses.jsonl", params.passes, make_accept(item_ids), progress)
    samples = sample_table(records, jobs, item_ids)
    scores = score_table(samples, item_ids)
    ok = samples[samples.ok]
    summary = {"questions_asked": len(ids), "calls": len(jobs), "answered": int(len(ok)), "failed": int((~samples.ok).sum()),
               "errors": samples.error.dropna().value_counts().head(10).to_dict(),
               "tokens": {k: int(ok[k].sum()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")},
               "tokens_per_call": {k: float(ok[k].mean()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")} if len(ok) else {},
               "items": len(item_ids), "params": params.model_dump(mode="json"), "backend": engine.describe(),
               "agreement": agreement(samples, item_ids),
               "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    example = (f"model: {params.model}\nsystem: {system}\nconfig (seed = sample index): {json.dumps(jobs[0].settings() | {'schema': SCHEMA})}\n\n"
               f"{jobs[0].prompt}") if jobs else ""
    return PresenceResult(samples, scores, summary, example)
