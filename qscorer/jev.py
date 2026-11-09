"""Checklist scoring with Jev, TypeSafe's System One model, through OpenRouter's decisions endpoint.

Jev does not generate text. A request gives it the content once (`state`: the question, its options, the correct answer
and the feedback, with the dataset's context sentence) and a set of typed questions; here one `noul` (yes/no) question per
checklist feature. It answers each with the probability that the answer is yes. A question is one request for all 45
features.

The instruction is the feature's criterion as a statement that is true when the feature is present ("Count as present
when the stem includes X" becomes "The stem includes X"), which is the form TypeSafe's Noul guide recommends. Two variants:
- `statement`: the statement alone ("The instruction is enough for most Nouls").
- `statement+definition`: the statement, with the feature's definition as the `true` criterion and its negation as `false`.
- `split`: `statement+definition` for every feature, plus each compound criterion in `configs/jev-splits.json` asked as
  separate parts (ids `f32.a`, `f32.b`, ...; the statement alone). The feature's score is the logical OR of its parts, the
  maximum part probability, which is at or above a threshold exactly when any part is. The joined question stays in the
  same request, and its probability is kept as `f32.joined`, so one run compares the split and the joined readings.
- `tuned`: the `split` request with the wordings chosen in `configs/jev-phrasings.json` (`_chosen`) in place of the
  joined question of those features, scored by `configs/jev-calibration-gemini.json` (per feature: joined or split form,
  then a threshold or a calibrated probability, fitted by `scripts/tune_jev.py` against Gemini). The raw probabilities stay
  in the scores as `<id>.joined` and, for split features, `<id>.split`.
The guide says to try both and keep whichever gives better answers.

The feature score is the probability. The regression was fitted on the share of Gemini calls that found a feature
present, so a probability sits on the same 0 to 1 scale. The guide does not endorse it as a graded score ("If the question
is really about degree, the value does not measure the degree") and suggests thresholds of 0.7 for yes and 0.3 for no, so
the comparison also reports the probability cut at 0.5 and 0.7. Jev takes no seed or temperature, so a question is asked once.

Endpoint: POST https://openrouter.ai/api/alpha/decisions with the OpenRouter key (alpha: the shape may change).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from qscorer.llm import ROOT, Backend, Job, api_key
from qscorer.presence import PresenceResult

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.14"
VARIANTS = ("statement", "statement+definition", "split", "tuned")
DEFAULT_VARIANT = "statement"
SPLITS = ROOT / "configs" / "jev-splits.json"
PHRASINGS = ROOT / "configs" / "jev-phrasings.json"
CALIBRATION = ROOT / "configs" / "jev-calibration-gemini.json"


def load_chosen(path: Path = PHRASINGS) -> dict[str, str]:
    """Feature id -> the wording the `tuned` variant sends."""
    d = json.loads(path.read_text())
    return {f: d[f][v] for f, v in d.get("_chosen", {}).items()}


def load_calibration(path: Path = CALIBRATION) -> dict[str, dict]:
    return json.loads(path.read_text())["features"]


def calibrated(choice: dict, joined: float, split: float | None) -> float:
    """One feature's score from its raw probabilities: the chosen form, then present-at-threshold or sigmoid(a*logit(p)+b)."""
    p = split if choice["form"] == "split" else joined
    if p is None:
        raise ValueError("the calibration chooses the split form for a feature with no split parts")
    if choice["mode"] == "threshold":
        return float(p >= choice["threshold"])
    x = math.log(min(max(p, 1e-3), 1 - 1e-3) / (1 - min(max(p, 1e-3), 1 - 1e-3)))
    return 1 / (1 + math.exp(-(choice["a"] * x + choice["b"])))


def load_splits(path: Path = SPLITS) -> dict[str, list[str]]:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def part_ids(fid: str, parts: list[str]) -> list[str]:
    return [f"{fid}.{chr(ord('a') + k)}" for k in range(len(parts))]


def state(q: dict, context: str) -> dict:
    return {"context": context, "question": q["question"]["stem"],
            "options": {c["label"]: c["text"] for c in q["question"]["choices"]},
            "correct_answer": q["answerKey"], "feedback": q.get("feedback-text", "") or ""}


def statement(criterion: str) -> str:
    """'Count as present when the stem includes X.' -> 'The stem includes X.'"""
    c = criterion.strip()
    for prefix in ("Count as present when ", "Count as present if ", "Present when "):
        if c.lower().startswith(prefix.lower()):
            c = c[len(prefix):]
            break
    c = c[:1].upper() + c[1:]
    return c if c.endswith(".") else c + "."


def questions_for(checklist: list[dict], variant: str = DEFAULT_VARIANT, splits: dict[str, list[str]] | None = None) -> dict:
    """The Noul questions of one request, keyed by the id each answer comes back under."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown Jev variant {variant!r}; one of {VARIANTS}")
    out = {}
    for it in checklist:
        q: dict = {"type": "noul", "instructions": statement(it["criterion"])}
        if variant in ("statement+definition", "split", "tuned"):
            q["criteria"] = {"true": f"{it['name']}: {it['definition'].strip()}",
                             "false": f"The question does not show this feature ({it['name']})."}
        out[it["id"]] = q
    if variant == "tuned":
        for fid, text in load_chosen().items():
            if fid in out:
                out[fid] = {"type": "noul", "instructions": text}
    if variant in ("split", "tuned"):
        splits = load_splits() if splits is None else splits
        known = {it["id"] for it in checklist}
        for fid, parts in splits.items():
            if fid not in known:
                raise ValueError(f"split for {fid}, which is not in the checklist")
            for pid, text in zip(part_ids(fid, parts), parts):
                out[pid] = {"type": "noul", "instructions": text}
    return out


def combine(p: dict[str, float], checklist: list[dict], variant: str, splits: dict[str, list[str]] | None = None) -> dict[str, float]:
    """Feature scores from one request's probabilities. For `split`: each split feature is the OR (maximum) of its parts,
    with the joined probability as `<id>.joined` and the parts under their own ids."""
    item_ids = [it["id"] for it in checklist]
    if variant == "tuned":
        splits = load_splits() if splits is None else splits
        cal = load_calibration()
        out = {}
        for i in item_ids:
            split = max(p[k] for k in part_ids(i, splits[i])) if i in splits else None
            out[i] = calibrated(cal[i], p[i], split)
            out[f"{i}.joined"] = p[i]
            if split is not None:
                out[f"{i}.split"] = split
                out.update({k: p[k] for k in part_ids(i, splits[i])})
        return out
    if variant != "split":
        return {i: p[i] for i in item_ids}
    splits = load_splits() if splits is None else splits
    out = {i: p[i] for i in item_ids}
    for fid, parts in splits.items():
        pids = part_ids(fid, parts)
        out[f"{fid}.joined"] = p[fid]
        out[fid] = max(p[k] for k in pids)
        out.update({k: p[k] for k in pids})
    return out


def build_jobs(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, model: str,
               variant: str = DEFAULT_VARIANT, splits: dict[str, list[str]] | None = None) -> list[Job]:
    """One job per question; the job's prompt is the request body as JSON (the decisions endpoint has no prompt)."""
    qs = questions_for(checklist, variant, splits)
    return [Job(f"{qid}|0", json.dumps({"model": model, "state": state(questions[qid], context), "questions": qs}),
                schema=None, thinking=None, thoughts=False) for qid in ids]


def nouls(raw: dict, item_ids: list[str]) -> dict[str, float] | None:
    """The probability for every feature, or None when the record is an error or any feature is missing or refused."""
    if raw.get("error"):
        return None
    answers = raw.get("answers") or {}
    out = {}
    for i in item_ids:
        a = answers.get(i) or {}
        p = a.get("noul")
        if a.get("type") != "noul" or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            return None
        out[i] = float(p)
    return out


class Jev(Backend):
    name = "openrouter-decisions"
    library = "httpx"

    def __init__(self, model: str = JEV_MODEL, key: str | None = None, concurrency: int = 16, delay: float = 0.02, timeout: float = 60.0):
        super().__init__(model, concurrency, delay, timeout)
        self.key = key or api_key("OPENROUTER_API_KEY")
        self._client = None
        self._loop = None

    def client(self):
        import asyncio
        import httpx
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout)
            self._loop = loop
        return self._client

    async def one(self, job: Job) -> dict:
        r = await self.client().post(DECISIONS_URL, content=job.prompt, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:500]}"}
        d = r.json()
        if not isinstance(d, dict) or d.get("error"):
            return {"error": str(d.get("error") if isinstance(d, dict) else d)[:500]}
        return d


def run_jev(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, out: Path,
            model: str = JEV_MODEL, variant: str = DEFAULT_VARIANT, concurrency: int = 16, timeout: float = 60.0, passes: int = 3,
            engine: Backend | None = None, splits: dict[str, list[str]] | None = None, progress=print) -> PresenceResult:
    """Score `ids` with Jev; raw log in `out/responses.jsonl`, resumed if present. Same result shape as `run_presence`;
    for `split` the scores carry the joined and part columns too."""
    item_ids = [it["id"] for it in checklist]
    if variant in ("split", "tuned") and splits is None:
        splits = load_splits()
    asked = list(questions_for(checklist, variant, splits))
    jobs = build_jobs(questions, ids, checklist, context, model, variant, splits)
    out.mkdir(parents=True, exist_ok=True)
    engine = engine or Jev(model, concurrency=concurrency, timeout=timeout)
    log = out / "responses.jsonl"
    records = engine.many(jobs, log, passes, lambda raw: nouls(raw, asked) is not None, progress)
    last_error = {}   # failed calls stay out of `records`; the log keeps their errors
    for line in log.read_text().splitlines() if log.exists() else []:
        if line.strip() and not nouls(rec := json.loads(line), asked):
            last_error[rec["id"]] = rec.get("error") or "incomplete answers"
    rows = []
    for job in jobs:
        qid = job.id.split("|")[0]
        raw = records.get(job.id)
        p = nouls(raw, asked) if raw else None
        p = combine(p, checklist, variant, splits) if p else None
        usage = (raw or {}).get("usage") or {}
        rows.append({"id": qid, "sample": 0, "ok": p is not None, **(p or {i: None for i in item_ids}),
                     "prompt_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "thought_tokens": 0,
                     "error": None if p else last_error.get(job.id, "not answered")})
    samples = pd.DataFrame(rows)
    ok = samples[samples.ok]
    extra = [c for c in samples.columns if "." in c]
    scores = ok[["id"] + item_ids + extra].assign(n_ok=1)[["id", "n_ok"] + item_ids + extra].reset_index(drop=True)
    summary = {"questions_asked": len(ids), "calls": len(jobs), "answered": int(len(ok)), "failed": int((~samples.ok).sum()),
               "tokens": {k: int(ok[k].fillna(0).sum()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")},
               "tokens_per_call": {k: float(ok[k].fillna(0).mean()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")} if len(ok) else {},
               "items": len(item_ids), "variant": variant, "backend": engine.describe(), "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    example = jobs[0].prompt if jobs else ""
    return PresenceResult(samples, scores, summary, example)
