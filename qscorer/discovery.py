"""Stage 2.2 (`discovery`): contrastive feature discovery, one batch and condition per call.

Each batch of the plan (`batches.csv`) is shown to Gemini under every condition its kind serves
(`CONDITIONS` in `qscorer/batches.py`): good batches under E1 and E2, bad batches under E3 and
E4, mixed batches under E5 to E10. The prompt is the design's Appendix A: the dataset's context
sentence, the four questions in the plan's order (each rendered as in Stage 1, headed
"Question 1" to "Question 4", with its reported HIGH-LEARNING or LOW-LEARNING label in the
labelled conditions, true or inverted), then prompt A, B, C or D by polarity and labels, and the
common output instructions. Good and bad features are always separate calls. Each batch and
condition is asked `samples` times with different seeds.

The reply is JSON: up to five features, each with a name, a definition, a list of questions
(1 to 4) each marked as exhibiting the feature or cited as a contrast where it is absent, with
exact words from the question as evidence, and a mechanism. Every evidence quote is checked
against the rendered text of the question it refers to (whitespace, case and bullet markers
ignored; a quote that stitches passages together, or prose with quoted segments, counts when
each piece is found). Two tables come out: `features.csv`, one row per feature per call, and
`attributions.csv`, one row per feature and listed question with its `exhibits` flag; the rows
with `exhibits` true are the design's Z_fjbc before feature normalisation (Stage 2.3).
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd

from qscorer.batches import CONDITIONS
from qscorer.config import DatasetConfig, DiscoveryParams
from qscorer.data import load_questions, render_question
from qscorer.llm import Job, make_backend, parse

PROMPTS = {
    ("good", False): (
        "Examine the four questions and their feedback. Identify up to five observable, generalizable features of these questions "
        "or their feedback that could facilitate improvement in students' accuracy over repeated practice with feedback.\n\n"
        "A feature may occur in one or more questions; it need not be shared by all four. Identify only features supported by "
        "evidence in the provided questions. Do not assume that these questions actually produced learning."),
    ("bad", False): (
        "Examine the four questions and their feedback. Identify up to five observable, generalizable features of these questions "
        "or their feedback that could impede improvement in students' accuracy over repeated practice with feedback.\n\n"
        "A feature may occur in one or more questions; it need not be shared by all four. Identify only features supported by "
        "evidence in the provided questions. Do not assume that these questions actually failed to produce learning."),
    ("good", True): (
        "Examine the four questions and their feedback. Each question has a reported HIGH-LEARNING or LOW-LEARNING label indicating "
        "whether students showed stronger or weaker improvement during repeated practice.\n\n"
        "Identify up to five observable, generalizable features that could help explain why HIGH-LEARNING questions show greater "
        "improvement than LOW-LEARNING questions.\n\n"
        "Focus only on potentially beneficial features. Use the LOW-LEARNING questions as contrasts. For each feature, identify which "
        "questions in both groups exhibit it, and provide supporting evidence.\n\n"
        "A feature need not appear in every HIGH-LEARNING question or be completely absent from LOW-LEARNING questions. Do not force "
        "distinctions if the examples do not support them."),
    ("bad", True): (
        "Examine the four questions and their feedback. Each question has a reported HIGH-LEARNING or LOW-LEARNING label indicating "
        "whether students showed stronger or weaker improvement during repeated practice.\n\n"
        "Identify up to five observable, generalizable features that could help explain why LOW-LEARNING questions show less "
        "improvement than HIGH-LEARNING questions.\n\n"
        "Focus only on potentially detrimental features. Use the HIGH-LEARNING questions as contrasts. For each feature, identify which "
        "questions in both groups exhibit it, and provide supporting evidence.\n\n"
        "A feature need not appear in every LOW-LEARNING question or be completely absent from HIGH-LEARNING questions. Do not force "
        "distinctions if the examples do not support them."),
}
OUTPUT = (
    "For each feature, provide:\n\n"
    "1. Name: a short, generalizable feature name.\n"
    "2. Definition: a precise description of the observable characteristic.\n"
    "3. Questions: which of the four questions exhibit the feature, each with exact words from its stem, answer options, or feedback "
    "supporting its presence. A question you cite as a contrast, where the feature is absent, is marked as not exhibiting it.\n"
    "4. Mechanism: one sentence explaining how the feature might affect improvement during repeated practice.\n\n"
    "Requirements:\n\n"
    "- Identify features of question design or feedback, not merely the science topic.\n"
    "- Do not propose features without concrete evidence.\n"
    "- Do not assume that a plausible learning mechanism is empirically established.\n"
    "- Return fewer than five features, or none, if the examples do not support additional features.\n"
    "- Prefer reusable features that can be evaluated consistently on new questions.\n\n"
    "Do not infer actual student outcomes beyond the information explicitly provided.\n\n"
    'Respond in JSON with one field, "features": a list of up to five objects, each with the fields "name", "definition", '
    '"questions" (a list of objects with "question", a number from 1 to 4; "exhibits", true if the question exhibits the feature and '
    'false if it is cited as a contrast; and "evidence", the exact words) and "mechanism".')
LABELLED_OUTPUT = "For each feature, report the questions in both groups that exhibit it, with supporting evidence from each.\n\n"
SCHEMA = {"type": "object", "required": ["features"], "properties": {"features": {"type": "array", "maxItems": 5, "items": {
    "type": "object", "required": ["name", "definition", "questions", "mechanism"],
    "properties": {"name": {"type": "string"}, "definition": {"type": "string"}, "mechanism": {"type": "string"},
                   "questions": {"type": "array", "items": {"type": "object", "required": ["question", "exhibits", "evidence"],
                                                           "properties": {"question": {"type": "integer"}, "exhibits": {"type": "boolean"},
                                                                          "evidence": {"type": "string"}}}}}}}}}
LABEL = {"high": "HIGH-LEARNING", "low": "LOW-LEARNING"}
FEATURE_COLUMNS = ["batch", "condition", "sample", "round", "set", "kind", "feature", "name", "definition", "mechanism", "questions", "n_questions",
                   "evidence_found"]
ATTRIBUTION_COLUMNS = ["batch", "condition", "sample", "round", "set", "kind", "feature", "name", "position", "id", "pool", "label", "exhibits", "evidence",
                       "evidence_found"]


@dataclass
class DiscoveryResult:
    calls: pd.DataFrame          # one row per call
    features: pd.DataFrame       # one row per feature per call
    attributions: pd.DataFrame   # one row per feature and question
    summary: dict
    example: str


def shown_label(pool: str, labels: str | None) -> str | None:
    """The label a question of `pool` carries under a condition's `labels` (None, true, inverted)."""
    if labels is None:
        return None
    return LABEL[pool] if labels == "true" else LABEL["low" if pool == "high" else "high"]


def build_prompt(members: pd.DataFrame, questions: dict[str, dict], condition: str, config: DatasetConfig) -> str:
    """The prompt for one batch (its plan rows, in position order) under one condition."""
    _, polarity, labels = CONDITIONS[condition]
    blocks = []
    for _, m in members.sort_values("position").iterrows():
        label = shown_label(m.pool, labels)
        blocks.append(render_question(questions[m.id], f"Question {m.position}" + (f" (reported label: {label})" if label else "")))
    intro = ("Below are four questions, each with its stem, four answer options, correct answer, and feedback"
             + (", and each question's reported label." if labels else "."))
    head = "\n\n".join([config.prompt.context, intro] + blocks)
    task = PROMPTS[polarity, labels is not None]
    return head + "\n\n\n" + task + "\n\n" + (LABELLED_OUTPUT if labels else "") + OUTPUT


def select_batches(plan: pd.DataFrame, params: DiscoveryParams) -> pd.DataFrame:
    """The plan rows asked: all, or with `limit` the first `limit` batches of each kind in the first round."""
    if params.limit is None:
        return plan
    first = plan[plan["round"] == plan["round"].min()]
    keep = [b for _, g in first.groupby("kind", sort=True) for b in sorted(g.batch.unique())[: params.limit]]
    return plan[plan.batch.isin(keep)]


def build_jobs(plan: pd.DataFrame, questions: dict[str, dict], config: DatasetConfig, params: DiscoveryParams) -> list[Job]:
    jobs = []
    for batch, members in plan.groupby("batch", sort=False):
        kind = members.kind.iloc[0]
        for condition, (c_kind, _, _) in CONDITIONS.items():
            if c_kind != kind:
                continue
            prompt = build_prompt(members, questions, condition, config)
            for k in range(params.samples):
                jobs.append(Job(f"{batch}|{condition}|{k}", prompt, system=config.prompt.system, schema=SCHEMA, temperature=params.temperature, seed=k,
                                max_output_tokens=params.max_output_tokens, thinking=params.thinking, thoughts=params.thoughts))
    return jobs


def normalise(text: str) -> str:
    """Text for the evidence check: bullet markers and the question header dropped, whitespace
    collapsed, case ignored."""
    text = re.sub(r"^(Question[^\n]*:|- )", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_found(quote: str, rendered: str) -> bool:
    """Whether the quote is in the question's rendered text. A quote that stitches passages
    together (sentences from the stem and the feedback, parts labelled "Stem:" or "Feedback:")
    counts when each of its fragments of at least 12 characters is found."""
    q = normalise(quote)
    if not q:
        return False
    if q in rendered:
        return True
    fragments = [f.strip() for f in re.split(r"(?:stem|feedback|options?|answer choices?|correct answer):|(?<=[.?!;])\s+|\.\.\.|\u2026", q)]
    fragments = [f for f in fragments if len(f) >= 12]
    if fragments and all(f in rendered for f in fragments):
        return True
    quoted = [m for m in re.findall(r"['\"\u2018\u201c]([^'\"\u2019\u201d]{12,})['\"\u2019\u201d]", q)]
    return bool(quoted) and all(m.strip() in rendered for m in quoted)


def parse_features(text: str | None) -> list[dict] | None:
    """The feature list from the JSON reply, with question numbers outside 1 to 4 dropped; None
    when the reply does not parse."""
    if not text:
        return None
    try:
        features = json.loads(text).get("features")
    except (ValueError, AttributeError):
        return None
    if not isinstance(features, list):
        return None
    out = []
    for f in features[:5]:
        if not isinstance(f, dict) or not isinstance(f.get("questions"), list):
            return None
        qs = [q for q in f["questions"] if isinstance(q, dict) and isinstance(q.get("question"), int) and 1 <= q["question"] <= 4]
        out.append({"name": str(f.get("name", "")), "definition": str(f.get("definition", "")), "mechanism": str(f.get("mechanism", "")),
                    "questions": [{"question": q["question"], "exhibits": bool(q.get("exhibits", True)), "evidence": str(q.get("evidence", ""))} for q in qs]})
    return out


def accept(raw: dict) -> bool:
    r = parse(raw)
    return r.ok and parse_features(r.text) is not None


def last_records(log: Path) -> dict[str, dict]:
    """The last raw record of every call in the log, accepted or not, so that a call that
    failed every pass reports why."""
    out: dict[str, dict] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                raw = json.loads(line)
                out[raw["id"]] = raw
    return out


def tables(records: dict[str, dict], jobs: list[Job], plan: pd.DataFrame, questions: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The call, feature and attribution tables. `records` holds the accepted raw record of
    each call, or its last failed one."""
    members = {b: g.sort_values("position") for b, g in plan.groupby("batch")}
    rendered = {q: normalise(render_question(questions[q])) for q in plan.id.unique()}
    calls, feats, attrs = [], [], []
    for job in jobs:
        batch, condition, k = job.id.split("|")
        m = members[batch]
        meta = {"batch": batch, "condition": condition, "sample": int(k), "round": int(m["round"].iloc[0]), "set": m.set.iloc[0], "kind": m.kind.iloc[0]}
        raw = records.get(job.id)
        r = parse(raw) if raw else None
        features = parse_features(r.text) if r else None
        calls.append(meta | {"n_features": None if features is None else len(features), "thought": r.thought if r else None,
                             "finish_reason": r.finish_reason if r else None, "prompt_tokens": r.prompt_tokens if r else None,
                             "output_tokens": r.output_tokens if r else None, "thought_tokens": r.thought_tokens if r else None,
                             "error": None if features is not None else (r.error or "reply does not parse") if r else "not asked"})
        if features is None:
            continue
        labels = CONDITIONS[condition][2]
        by_position = {int(row.position): row for row in m.itertuples()}
        for i, f in enumerate(features, 1):
            found = []
            for q in f["questions"]:
                row = by_position[q["question"]]
                ok = evidence_found(q["evidence"], rendered[row.id])
                found.append(ok)
                attrs.append(meta | {"feature": i, "name": f["name"], "position": q["question"], "id": row.id, "pool": row.pool,
                                     "label": shown_label(row.pool, labels), "exhibits": q["exhibits"], "evidence": q["evidence"], "evidence_found": ok})
            shown = [q["question"] for q in f["questions"] if q["exhibits"]]
            feats.append(meta | {"feature": i, "name": f["name"], "definition": f["definition"], "mechanism": f["mechanism"],
                                 "questions": " ".join(str(q) for q in shown), "n_questions": len(shown),
                                 "evidence_found": (sum(found) / len(found)) if found else None})
    return pd.DataFrame(calls), pd.DataFrame(feats, columns=FEATURE_COLUMNS), pd.DataFrame(attrs, columns=ATTRIBUTION_COLUMNS)


def summarise(calls: pd.DataFrame, features: pd.DataFrame, attributions: pd.DataFrame, config: DatasetConfig, params: DiscoveryParams, jobs: list[Job]) -> dict:
    ok = calls[calls.n_features.notna()]
    per_condition = {}
    for c, g in ok.groupby("condition", sort=False):
        f = features[features.condition == c]
        listed = attributions[attributions.condition == c]
        a = listed[listed.exhibits]
        entry = {"calls": int(len(g)), "features_per_call": float(g.n_features.mean()), "questions_per_feature": float(f.n_questions.mean()) if len(f) else None,
                 "contrasts_per_feature": float((~listed.exhibits).sum() / len(f)) if len(f) else None,
                 "evidence_found": float(listed.evidence_found.mean()) if len(listed) else None}
        if CONDITIONS[c][0] == "mixed" and len(a):
            entry["attributions_to_high_members"] = float((a.pool == "high").mean())
            if CONDITIONS[c][2]:
                entry["attributions_to_HIGH_labelled"] = float((a.label == "HIGH-LEARNING").mean())
        per_condition[c] = entry
    return {
        "dataset": config.name, "params": params.model_dump(mode="json"),
        "batches_asked": int(calls.batch.nunique()), "calls": len(jobs), "answered": int(len(ok)), "failed": int(calls.n_features.isna().sum()),
        "errors": calls.error.dropna().value_counts().head(10).to_dict(),
        "tokens": {k: int(ok[k].sum()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")},
        "tokens_per_call": {k: float(ok[k].mean()) for k in ("prompt_tokens", "output_tokens", "thought_tokens")} if len(ok) else {},
        "features_per_call": ok.n_features.value_counts().sort_index().to_dict() if len(ok) else {},
        "evidence_found": float(attributions.evidence_found.mean()) if len(attributions) else None,
        "per_condition": per_condition,
        "versions": {"python": platform.python_version(), "pandas": metadata.version("pandas")},
        "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_discovery(config: DatasetConfig, params: DiscoveryParams, plan: pd.DataFrame, out: Path, engine=None, progress=print) -> DiscoveryResult:
    """Stage 2.2 end to end: choose the batches, ask Gemini (raw log in `out/responses.jsonl`,
    resumed if present), tabulate. `engine` replaces the Gemini client in tests."""
    questions = load_questions(config.questions)
    plan = select_batches(plan, params)
    jobs = build_jobs(plan, questions, config, params)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "responses.jsonl"
    engine = engine or make_backend(params)
    records = engine.many(jobs, log, params.passes, accept, progress)
    calls, features, attributions = tables(last_records(log) | records, jobs, plan, questions)
    summary = summarise(calls, features, attributions, config, params, jobs)
    summary["backend"] = engine.describe()
    examples = []
    for condition in ("E1", "E7"):
        job = next((j for j in jobs if j.id.split("|")[1] == condition), None)
        if job:
            examples.append(f"=== {job.id} ===\nmodel: {params.model}\nsystem: {config.prompt.system}\nconfig (seed = sample index): "
                            f"{json.dumps(job.settings())}\n\n{job.prompt}")
    return DiscoveryResult(calls, features, attributions, summary, "\n\n".join(examples))
