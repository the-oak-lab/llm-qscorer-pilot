"""Stage 3.2 with Jev (`jev`): score questions against the checklist with TypeSafe's Jev instead of Gemini.

Jev does not generate text. A request gives it the content once (the state: the dataset's context sentence, the stem, the
options, the correct answer and the feedback) and one yes/no question (a Noul) per checklist item, and Jev answers each with
the probability that the answer is yes. One request covers the whole checklist. Jev takes no seed or temperature, but the
same request asked again gives slightly different probabilities (by about 0.01 on average, up to about 0.1), so a question
is asked `samples` times (`SAMPLES`, 3, by default) and its score on an item is the mean of the probabilities. The repeats
of a question are sent apart in time: all questions once, then all again.

Each item is asked as its criterion turned into a statement that is true when the item is present ("Count as present when
the stem includes X" becomes "The stem includes X"), with the item's definition as what yes means. A few items are asked in
other words (`configs/jev-rewordings.json`), chosen on the `jev` branch (Dallas Hall) where Jev read the statement
differently from Gemini. An item's **score** is the mean probability. The slope regression is fitted on Jev's own scores, since
they do not mean the same as Gemini's share of calls. For the flaws, an item counts as present when its mean probability is
at least `PRESENT`, 0.7, the threshold TypeSafe suggests for yes.

Jev is reached in one of two ways, with the same request and answer: TypeSafe's API (`direct`, TYPESAFE_API_KEY) or
OpenRouter's decisions endpoint (`openrouter`, OPENROUTER_API_KEY; OpenRouter marks it alpha). The version is pinned.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx
import pandas as pd

from qscorer.llm import ROOT, Backend, Job, api_key
from qscorer.presence import PresenceResult

PRESENT = 0.7
SAMPLES = 3
MODEL = "jev-1.14.0"
OPENROUTER_IDS = {"jev-1.14.0": "typesafe/jev-1.14"}   # TypeSafe's name of a version -> OpenRouter's
APIS = {"direct": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY")}
REWORDINGS = ROOT / "configs" / "jev-rewordings.json"


def model_id(model: str, api: str) -> str:
    """A Jev version, given by either API's name, as `api` names it."""
    direct = {v: k for k, v in OPENROUTER_IDS.items()}.get(model, model)
    if api == "direct":
        return direct
    if direct not in OPENROUTER_IDS:
        raise ValueError(f"no OpenRouter name known for {model}; give the model id yourself")
    return OPENROUTER_IDS[direct]


def load_rewordings(path: Path = REWORDINGS) -> dict[str, str]:
    """Item id -> the statement asked instead of the item's own."""
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def statement(criterion: str) -> str:
    """'Count as present when the stem includes X.' -> 'The stem includes X.'"""
    c = criterion.strip()
    for prefix in ("Count as present when ", "Count as present if ", "Present when "):
        if c.lower().startswith(prefix.lower()):
            c = c[len(prefix):]
            break
    c = c[:1].upper() + c[1:]
    return c if c.endswith(".") else c + "."


def questions_for(checklist: list[dict], rewordings: dict[str, str]) -> dict:
    """One Noul per item, keyed by item id: the criterion as a statement with the definition as what yes means, or the item's
    rewording alone."""
    unknown = set(rewordings) - {it["id"] for it in checklist}
    if unknown:
        raise ValueError(f"rewordings for items not in the checklist: {sorted(unknown)}")
    out = {}
    for it in checklist:
        if it["id"] in rewordings:
            out[it["id"]] = {"type": "noul", "instructions": rewordings[it["id"]]}
        else:
            out[it["id"]] = {"type": "noul", "instructions": statement(it["criterion"]),
                             "criteria": {"true": f"{it['name']}: {it['definition'].strip()}",
                                          "false": f"The question does not show this feature ({it['name']})."}}
    return out


def state(q: dict, context: str) -> dict:
    return {"context": context, "question": q["question"]["stem"], "options": {c["label"]: c["text"] for c in q["question"]["choices"]},
            "correct_answer": q["answerKey"], "feedback": q.get("feedback-text", "") or ""}


def build_jobs(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, rewordings: dict[str, str], model: str,
               samples: int = 1) -> list[Job]:
    """`samples` jobs per question (ids `<question>|<k>`), every question's k-th before any (k+1)-th; a job's prompt is the
    request body as JSON (Jev takes no prompt)."""
    asked = questions_for(checklist, rewordings)
    bodies = {qid: json.dumps({"model": model, "state": state(questions[qid], context), "questions": asked}) for qid in ids}
    return [Job(f"{qid}|{k}", bodies[qid], thinking=None, thoughts=False) for k in range(samples) for qid in ids]


def nouls(raw: dict, item_ids: list[str]) -> dict[str, float] | None:
    """The probability for every item, or None when the record is an error or an item is missing or not a probability."""
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
    library = "httpx"

    def __init__(self, api: str = "direct", model: str = MODEL, concurrency: int = 16, delay: float = 0.02, timeout: float = 60.0):
        super().__init__(model_id(model, api), concurrency, delay, timeout)
        self.name = f"jev-{api}"
        self.url, key = APIS[api]
        self.key = api_key(key)
        self._client, self._loop = None, None

    def client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client, self._loop = httpx.AsyncClient(headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout), loop
        return self._client

    async def one(self, job: Job) -> dict:
        r = await self.client().post(self.url, content=job.prompt, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:500]}"}
        d = r.json()
        if not isinstance(d, dict) or d.get("error"):
            return {"error": str(d.get("error") if isinstance(d, dict) else d)[:500]}
        return d


def run_jev(questions: dict[str, dict], ids: list[str], checklist: list[dict], context: str, rewordings: dict[str, str], out: Path,
            api: str = "direct", model: str = MODEL, samples: int = SAMPLES, concurrency: int = 16, timeout: float = 60.0, passes: int = 3,
            engine=None, progress=print) -> PresenceResult:
    """Score `ids` with Jev, `samples` calls per question; raw log in `out/responses.jsonl`, resumed if present (so a run with more
    samples adds only the new calls). The result has the shape of `run_presence`'s; a score is the mean probability over a
    question's answered calls."""
    item_ids = [it["id"] for it in checklist]
    engine = engine or Jev(api, model, concurrency=concurrency, timeout=timeout)
    jobs = build_jobs(questions, ids, checklist, context, rewordings, engine.model, samples)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "responses.jsonl"
    records = engine.many(jobs, log, passes, lambda raw: nouls(raw, item_ids) is not None, progress)
    errors = {}   # failed calls stay out of `records`; the log keeps their last error
    for line in log.read_text().splitlines() if log.exists() else []:
        if line.strip() and nouls(rec := json.loads(line), item_ids) is None:
            errors[rec["id"]] = rec.get("error") or "incomplete answer"
    rows = []
    for job in jobs:
        raw = records.get(job.id) or {}
        p = nouls(raw, item_ids) if raw else None
        usage = raw.get("usage") or {}
        qid, k = job.id.split("|")
        rows.append({"id": qid, "sample": int(k), "ok": p is not None, **(p or dict.fromkeys(item_ids)), "served_by": raw.get("model"),
                     "prompt_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                     "error": None if p else errors.get(job.id, "not asked")})
    calls = pd.DataFrame(rows)
    ok = calls[calls.ok]
    g = ok.groupby("id", sort=False)
    scores = g[item_ids].mean().assign(n_ok=g.size()).reset_index()[["id", "n_ok"] + item_ids]
    summary = {"questions_asked": len(ids), "calls": len(jobs), "answered": len(ok), "failed": int((~calls.ok).sum()),
               "errors": calls.error.dropna().value_counts().head(10).to_dict(), "served_by": ok.served_by.value_counts().to_dict(),
               "tokens": {k: int(ok[k].fillna(0).sum()) for k in ("prompt_tokens", "output_tokens")}, "items": len(item_ids),
               "params": {"engine": "jev", "api": api, "model": engine.model, "samples": samples, "rewordings": rewordings, "present": PRESENT},
               "repeat_difference": repeat_difference(ok, item_ids), "backend": engine.describe(),
               "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    return PresenceResult(calls, scores, summary, jobs[0].prompt if jobs else "")


def repeat_difference(ok: pd.DataFrame, item_ids: list[str]) -> dict | None:
    """For questions answered by calls 0 and 1: the mean and the largest absolute difference of their probabilities, over all items."""
    a, b = ok[ok["sample"] == 0].set_index("id"), ok[ok["sample"] == 1].set_index("id")
    both = a.index.intersection(b.index)
    if not len(both):
        return None
    d = (a.loc[both, item_ids] - b.loc[both, item_ids]).abs().to_numpy()
    return {"questions": len(both), "mean": float(d.mean()), "max": float(d.max())}
