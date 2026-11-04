"""Stage 1 (`baseline`): Gemini's direct, no-checklist learning predictions.

Each question is shown alone (stem, four options, correct answer, feedback, and the dataset's
context sentence) and Gemini is asked, in separate calls, the design's question (how many of 100
students will improve in accuracy on it with practice) and a second question that carried signal in
earlier work (how many of 100 students already know the answer before practice). Each question is
asked `samples` times per prompt with different seeds; the answer comes
back as JSON (`reason`, `answer`), and the thought summary and token counts of every call are
kept. A question's **rating** is the mean answer for `improve` and 100 minus the mean answer for
`known`, so that a higher rating always means more predicted learning.

Evaluation against Stage 0: Spearman correlation of the rating with the posterior slope (overall,
with a bootstrap interval, and within item sets), the same after partialling out first-practice
accuracy, the AUC of high- versus low-learning questions among the kept pairs, the share of kept
pairs whose high member gets the higher rating, and the consistency of the answers over samples
(a one-way intraclass correlation and the number of samples needed for a reliability of 0.90).
"""
from __future__ import annotations

import datetime as dt
import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from qscorer.config import BaselineParams, DatasetConfig
from qscorer.data import load_questions, render_question
from qscorer.llm import Job, make_backend, parse

TASKS = {
    "improve": ("Your task:\nGiven the features of the question and its feedback, estimate how many students (out of 100) will "
                "show improvement in accuracy on this question with practice. Give a short reason, then your estimate as a whole number."),
    "known": ("Your task:\nGiven the features of the question and its feedback, estimate how many students (out of 100) already "
              "know the correct answer to this question before practice. Give a short reason, then your estimate as a whole number."),
}
FORMAT = 'Respond in JSON with two fields: "reason" (one or two sentences) and "answer" (a whole number from 0 to 100).'
SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "answer": {"type": "integer"}}, "required": ["reason", "answer"]}


@dataclass
class BaselineResult:
    samples: pd.DataFrame     # one row per call
    ratings: pd.DataFrame     # one row per question and prompt
    evaluation: dict
    summary: dict
    example: str


def build_prompt(q: dict, task: str, config: DatasetConfig) -> str:
    head = "\n\n".join([config.prompt.context, "Below is one question with its stem, four answer options, correct answer, and feedback.",
                        render_question(q)])
    return head + "\n\n\n" + TASKS[task] + "\n\n" + FORMAT  # two blank lines set the task apart from the feedback


def select_questions(questions: dict[str, dict], config: DatasetConfig, params: BaselineParams, sets: pd.Series | None) -> list[str]:
    """The ids asked: the bank minus `config.exclude`; with `limit`, that many spread over the
    item sets in turn (shuffled within set by `seed`)."""
    ids = [q for q in questions if q not in set(config.exclude)]
    if params.limit is None or params.limit >= len(ids):
        return ids
    rng = np.random.default_rng(params.seed)
    by_set: dict[str, list[str]] = {}
    for q in ids:
        by_set.setdefault(sets.get(q, "set-1") if sets is not None else "set-1", []).append(q)
    queues = [list(rng.permutation(v)) for _, v in sorted(by_set.items())]
    out: list[str] = []
    while len(out) < params.limit:
        for queue in queues:
            if queue and len(out) < params.limit:
                out.append(queue.pop())
    return out


def build_jobs(questions: dict[str, dict], ids: list[str], config: DatasetConfig, params: BaselineParams) -> list[Job]:
    return [Job(f"{qid}|{task}|{k}", build_prompt(questions[qid], task, config), system=config.prompt.system, schema=SCHEMA, temperature=params.temperature,
                seed=k, max_output_tokens=params.max_output_tokens, thinking=params.thinking, thoughts=params.thoughts)
            for qid in ids for task in params.prompts for k in range(params.samples)]


def parse_answer(text: str | None) -> tuple[int | None, str | None]:
    """The integer answer and the reason from the JSON reply; (None, None) when it does not parse."""
    if not text:
        return None, None
    try:
        d = json.loads(text)
        answer = d.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not 0 <= answer <= 100:
            return None, d.get("reason")
        return int(round(answer)), d.get("reason")
    except (ValueError, AttributeError):
        return None, None


def accept(raw: dict) -> bool:
    """A record worth keeping: no error and an answer that parses."""
    r = parse(raw)
    return r.ok and parse_answer(r.text)[0] is not None


def sample_table(records: dict[str, dict], jobs: list[Job]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        qid, task, k = job.id.split("|")
        raw = records.get(job.id)
        r = parse(raw) if raw else None
        answer, reason = parse_answer(r.text) if r else (None, None)
        rows.append({"id": qid, "prompt": task, "sample": int(k), "answer": answer,
                     "rating": None if answer is None else (100 - answer if task == "known" else answer),
                     "reason": reason, "thought": r.thought if r else None, "finish_reason": r.finish_reason if r else None,
                     "prompt_tokens": r.prompt_tokens if r else None, "output_tokens": r.output_tokens if r else None,
                     "thought_tokens": r.thought_tokens if r else None, "error": (r.error if r else "not asked")})
    return pd.DataFrame(rows)


def rating_table(samples: pd.DataFrame) -> pd.DataFrame:
    g = samples.dropna(subset=["answer"]).groupby(["id", "prompt"])
    t = pd.DataFrame({"n_ok": g.size(), "answer": g.answer.mean(), "answer_sd": g.answer.std(ddof=1), "rating": g.rating.mean()})
    return t.reset_index()


def consistency(samples: pd.DataFrame, target: float = 0.90) -> dict:
    """Per prompt: the one-way intraclass correlation of a single answer (between-question variance
    over total), the reliability of the mean over the samples actually taken, and the samples
    needed for a mean with reliability `target` (Spearman-Brown)."""
    out = {}
    for task, s in samples.dropna(subset=["answer"]).groupby("prompt"):
        counts = s.groupby("id").size()
        if len(counts) < 3 or counts.max() < 2:
            out[task] = None
            continue
        k = float(counts.mean())
        means = s.groupby("id").answer.mean()
        grand = s.answer.mean()
        msb = float((counts * (means - grand) ** 2).sum() / (len(counts) - 1))
        msw = float(((s.answer - s.id.map(means)) ** 2).sum() / (len(s) - len(counts)))
        icc1 = (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw > 0 else float("nan")
        icc_mean = k * icc1 / (1 + (k - 1) * icc1) if np.isfinite(icc1) and icc1 > -1 / (k - 1) else float("nan")
        needed = target * (1 - icc1) / (icc1 * (1 - target)) if np.isfinite(icc1) and icc1 > 0 else float("inf")
        out[task] = {"questions": int(len(counts)), "samples_per_question": k, "icc_single": icc1, "reliability_of_mean": icc_mean,
                     "samples_for_0.90": float(max(1, np.ceil(needed))) if np.isfinite(needed) else None,
                     "sd_between_questions": float(means.std(ddof=1)), "sd_within_question": float(np.sqrt(msw))}
    return out


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Spearman correlation of x and y after regressing both ranks on the rank of z."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def evaluate(ratings: pd.DataFrame, slopes: pd.DataFrame | None, pairs_questions: pd.DataFrame | None, seed: int = 0, boot: int = 2000) -> dict:
    """Each prompt's rating against the Stage 0 slope and the Stage 0.2 labels."""
    out: dict = {}
    if slopes is None:
        return {"note": "no Stage 0 table; run scripts/run_slopes.py first"}
    s = slopes.set_index("id")
    labels = pairs_questions.set_index("id") if pairs_questions is not None else None
    rng = np.random.default_rng(seed)
    for task, r in ratings.groupby("prompt"):
        t = r.set_index("id").join(s[["set", "slope", "first_practice_accuracy"]], how="inner").dropna(subset=["slope", "rating"])
        e: dict = {"questions": int(len(t))}
        if len(t) >= 5:
            rho = float(spearmanr(t.rating, t.slope)[0])
            bs = [spearmanr(t.rating.to_numpy()[i], t.slope.to_numpy()[i])[0] for i in (rng.integers(0, len(t), len(t)) for _ in range(boot))]
            e |= {"spearman": rho, "spearman_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                  "partial_given_first_practice_accuracy": partial_spearman(t.rating, t.slope, t.first_practice_accuracy.fillna(t.first_practice_accuracy.mean()))}
            within = {k: (float(spearmanr(g.rating, g.slope)[0]), len(g)) for k, g in t.groupby("set") if len(g) >= 5}
            e["within_sets"] = {k: v[0] for k, v in within.items()}
            e["within_sets_weighted"] = float(sum(v[0] * v[1] for v in within.values()) / sum(v[1] for v in within.values())) if within else None
        if labels is not None:
            lab = t.join(labels[["pool", "partner"]], how="inner")
            kept = lab[lab.partner.fillna("").ne("")]
            hi, lo = kept[kept.pool == "high"].rating, kept[kept.pool == "low"].rating
            if len(hi) and len(lo):
                e["auc_high_vs_low"] = float((hi.to_numpy()[:, None] > lo.to_numpy()[None, :]).mean() + 0.5 * (hi.to_numpy()[:, None] == lo.to_numpy()[None, :]).mean())
                pairs = kept[kept.pool == "high"]
                both = pairs[pairs.partner.isin(kept.index)]
                if len(both):
                    hr, lr = both.rating.to_numpy(), kept.rating[both.partner].to_numpy()
                    e["kept_pairs_with_both_rated"] = int(len(both))
                    e["high_member_rated_higher"] = float(((hr > lr) + 0.5 * (hr == lr)).mean())
        out[task] = e
    return out


def run_baseline(config: DatasetConfig, params: BaselineParams, out: Path, slopes: pd.DataFrame | None = None,
                 pairs_questions: pd.DataFrame | None = None, engine=None, progress=print) -> BaselineResult:
    """Stage 1 end to end: choose the questions, ask Gemini (raw log in `out/responses.jsonl`,
    resumed if present), tabulate, evaluate. `engine` replaces the Gemini client in tests: any
    object with `many(jobs, log, passes, accept, progress)` running synchronously."""
    questions = load_questions(config.questions)
    sets = slopes.set_index("id").set if slopes is not None and "set" in slopes.columns else None
    ids = select_questions(questions, config, params, sets)
    jobs = build_jobs(questions, ids, config, params)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "responses.jsonl"
    engine = engine or make_backend(params)
    records = engine.many(jobs, log, params.passes, accept, progress)
    samples = sample_table(records, jobs)
    ratings = rating_table(samples)
    evaluation = evaluate(ratings, slopes, pairs_questions, params.seed)
    ok = samples.dropna(subset=["answer"])
    summary = {
        "dataset": config.name, "params": params.model_dump(mode="json"),
        "questions_asked": len(ids), "calls": len(jobs), "answered": int(len(ok)), "failed": int(samples.answer.isna().sum()),
        "errors": samples.error.dropna().value_counts().head(10).to_dict(),
        "tokens": {k: int(ok[k].sum()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")},
        "tokens_per_call": {k: float(ok[k].mean()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")} if len(ok) else {},
        "answers": {task: {"mean": float(g.answer.mean()), "sd_between_questions": float(g.groupby("id").answer.mean().std(ddof=1))}
                    for task, g in ok.groupby("prompt")},
        "consistency": consistency(samples),
        "backend": engine.describe(), "versions": {"python": platform.python_version(), "pandas": metadata.version("pandas")},
        "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    example = "\n\n".join(f"=== {task} ===\nmodel: {params.model}\nsystem: {config.prompt.system}\nconfig (seed = sample index): "
                          f"{json.dumps(jobs[0].settings() | {'schema': SCHEMA})}\n\n"
                          f"{build_prompt(questions[ids[0]], task, config)}" for task in params.prompts)
    return BaselineResult(samples, ratings, evaluation, summary, example)
