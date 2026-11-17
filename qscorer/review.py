"""Teacher review: turn the checklist scores of new questions into flags a teacher can act on.

The checklist items fall into three groups, set by hand in a review file next to the model
(`models/<name>/review.json`): **flaws**, known item-writing or feedback problems such as a correct
option longer than the others; **features**, what the question does, described without a
judgement; and **hidden** items, not shown. The groups follow item-writing guidance, not the
signs of the fitted coefficients, because several coefficients reward flaws (a longer correct
option predicts a steeper practice slope: students learn the cue).

A question's flaw is a candidate when its Stage 3.2 score is at least `threshold`. One more call
per question with candidates checks each one again, and for those that apply quotes the words
that show it, explains it in plain language and suggests one revision. The quote is checked
against the question text. Features present at the threshold are listed by part (stem, options,
feedback). The predicted slope is not part of the review.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from qscorer.config import PresenceParams
from qscorer.data import render_question
from qscorer.llm import Job, make_backend, parse
from qscorer.presence import render_checklist

TASK = ("Your task:\nBelow are possible problems that questions like this one sometimes have. The teacher who wrote this question will read your notes, "
        "so flag a problem only when it clearly applies. For each problem, check the question against its definition and criterion. "
        "When it applies, quote the exact words from the stem, options or feedback that show it (copied verbatim, at most about 25 words), "
        "explain the problem in one or two plain sentences a teacher can act on, and suggest one concrete revision. "
        "When it does not apply, say why in the explanation and leave the quote and the suggestion empty.")
FORMAT = ('Respond in JSON with one field, "flags": a list with one object per problem, in the order given, each with "id" (the problem\'s id), '
          '"applies" (true or false), "quote", "explanation" and "suggestion" (strings).')
FIELDS = ("quote", "explanation", "suggestion")
SCHEMA = {"type": "object", "required": ["flags"], "properties": {"flags": {"type": "array", "items": {
    "type": "object", "required": ["id", "applies", *FIELDS],
    "properties": {"id": {"type": "string"}, "applies": {"type": "boolean"}, **{k: {"type": "string"} for k in FIELDS}}}}}}
PARTS = {"stem": "Stem", "options": "Options", "feedback": "Feedback"}


def load_groups(path: Path) -> dict[str, dict]:
    """item id -> {group, label, why}, from a review file."""
    return {it["id"]: it for it in json.load(open(path))["items"]}


def build_prompt(q: dict, candidates: list[dict], context: str) -> str:
    head = "\n\n".join([context, "Below is one question with its stem, four answer options, correct answer, and feedback.", render_question(q)])
    return head + "\n\n\n" + TASK + "\n\nPossible problems:\n" + render_checklist(candidates) + "\n\n" + FORMAT


def parse_review(text: str | None, ids: list[str]) -> dict[str, dict] | None:
    """item id -> {applies, quote, explanation, suggestion}, when every candidate is answered; else None."""
    if not text:
        return None
    try:
        flags = json.loads(text).get("flags")
    except (ValueError, AttributeError):
        return None
    if not isinstance(flags, list):
        return None
    out = {}
    for f in flags:
        if isinstance(f, dict) and isinstance(f.get("applies"), bool) and str(f.get("id")).strip() in ids:
            out.setdefault(str(f["id"]).strip(), {"applies": f["applies"], **{k: str(f.get(k) or "").strip() for k in FIELDS}})
    return out if all(i in out for i in ids) else None


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’]", "", text)).strip().lower()


def quote_found(quote: str, q: dict) -> bool:
    """Whether the quote, or each part of it split at an ellipsis, appears in the question, ignoring case, quotes and spacing."""
    whole = squash(render_question(q))
    parts = [squash(p) for p in re.split(r"\.\.\.|…", quote) if squash(p)]
    return bool(parts) and all(p in whole for p in parts)


def candidates(scores: pd.DataFrame, groups: dict[str, dict], threshold: float) -> dict[str, list[str]]:
    """question id -> the flaw items scored at or above `threshold`."""
    flaws = [i for i, g in groups.items() if g["group"] == "flaw" and i in scores.columns]
    s = scores.set_index("id")
    return {qid: [i for i in flaws if s.at[qid, i] >= threshold] for qid in s.index}


def run_review(questions: dict[str, dict], scores: pd.DataFrame, checklist: list[dict], groups: dict[str, dict], context: str, system: str,
               params: PresenceParams, log: Path, threshold: float = 0.5, engine=None, progress=print) -> pd.DataFrame:
    """Check every candidate flaw with one call per question; raw log in `log`, resumed if present.
    One row per question and candidate: id, item, checked (the call answered), applies, quote, quote_found, explanation, suggestion."""
    items = {it["id"]: it for it in checklist}
    todo = {qid: ids for qid, ids in candidates(scores, groups, threshold).items() if ids}
    jobs = [Job(qid, build_prompt(questions[qid], [items[i] for i in ids], context), system=system, schema=SCHEMA, temperature=params.temperature,
                seed=0, max_output_tokens=params.max_output_tokens, thinking=params.thinking, thoughts=params.thoughts) for qid, ids in todo.items()]
    def accept(raw: dict) -> bool:
        r = parse(raw)
        return r.ok and parse_review(r.text, todo[r.id]) is not None
    log.parent.mkdir(parents=True, exist_ok=True)
    engine = engine or make_backend(params)
    records = engine.many(jobs, log, params.passes, accept, progress) if jobs else {}
    rows = []
    for qid, ids in todo.items():
        r = parse(records[qid]) if qid in records else None
        answers = parse_review(r.text, ids) if r else None
        for i in ids:
            a = (answers or {}).get(i)
            rows.append({"id": qid, "item": i, "checked": a is not None, "applies": a["applies"] if a else None,
                         "quote": a["quote"] if a else "", "quote_found": quote_found(a["quote"], questions[qid]) if a and a["quote"] else None,
                         "explanation": a["explanation"] if a else "", "suggestion": a["suggestion"] if a else ""})
    return pd.DataFrame(rows, columns=["id", "item", "checked", "applies", "quote", "quote_found", "explanation", "suggestion"])


def build_report(questions: dict[str, dict], scores: pd.DataFrame, flags: pd.DataFrame, checklist: list[dict], groups: dict[str, dict],
                 threshold: float = 0.5) -> list[dict]:
    """One entry per scored question, most flaws first: its flaws (confirmed, or unchecked when the check call failed) and its features by part."""
    parts = {it["id"]: it["part"] for it in checklist}
    s = scores.set_index("id")
    report = []
    for qid in s.index:
        f = flags[flags.id == qid]
        flaws = [{"id": r.item, "label": groups[r.item]["label"], "why": groups[r.item].get("why", ""), "checked": bool(r.checked),
                  "quote": r.quote, "quote_found": r.quote_found, "explanation": r.explanation, "suggestion": r.suggestion}
                 for r in f.itertuples() if r.applies or not r.checked]
        features = {p: [groups[i]["label"] for i in parts if parts[i] == p and groups.get(i, {}).get("group") == "feature" and s.at[qid, i] >= threshold]
                    for p in PARTS}
        report.append({"id": qid, "flaws": flaws, "features": features})
    order = {qid: k for k, qid in enumerate(questions)}
    return sorted(report, key=lambda e: (-len(e["flaws"]), order.get(e["id"], 0)))


CSS = """
:root { --bg: #fbfaf7; --card: #ffffff; --ink: #1f2328; --muted: #5d6470; --line: #e3e0d8; --flag: #b4532a; --flag-bg: #fbeee6; --chip: #eef1f4; --key: #2f6f4f; }
@media (prefers-color-scheme: dark) { :root { --bg: #16181c; --card: #1f2227; --ink: #e8e6e1; --muted: #a2a8b2; --line: #33373e; --flag: #f0a07a; --flag-bg: #3a2a22; --chip: #2a2e35; --key: #7fc4a0; } }
body { background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; margin: 0; padding: 24px 16px; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 8px; }
.note { color: var(--muted); margin: 0 0 24px; }
article { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px 20px; margin: 0 0 20px; }
article h2 { font-size: 1.05rem; margin: 0 0 12px; display: flex; justify-content: space-between; gap: 12px; }
.count { color: var(--flag); font-weight: 600; white-space: nowrap; }
.count.none { color: var(--muted); font-weight: 400; }
.question { border-left: 3px solid var(--line); padding-left: 12px; margin-bottom: 16px; }
.question ol { list-style: none; padding: 0; margin: 8px 0; }
.question li.key { color: var(--key); font-weight: 600; }
.feedback { color: var(--muted); font-size: 0.95rem; white-space: pre-line; }
.flaw { background: var(--flag-bg); border-radius: 6px; padding: 10px 14px; margin: 0 0 10px; }
.flaw h3 { font-size: 1rem; margin: 0 0 4px; color: var(--flag); }
.flaw blockquote { margin: 6px 0; padding-left: 10px; border-left: 2px solid var(--flag); font-style: italic; }
.flaw p { margin: 4px 0; }
.why { color: var(--muted); font-size: 0.9rem; }
.features { margin-top: 12px; font-size: 0.92rem; }
.features div { margin: 4px 0; }
.features b { color: var(--muted); font-weight: 600; margin-right: 6px; }
.chip { display: inline-block; background: var(--chip); border-radius: 12px; padding: 1px 10px; margin: 2px 4px 2px 0; }
"""


def render_html(report: list[dict], questions: dict[str, dict], title: str = "Question review") -> str:
    e = html.escape
    n_flagged = sum(1 for r in report if r["flaws"])
    cards = []
    for r in report:
        q = questions[r["id"]]
        opts = "".join(f'<li class="{"key" if c["label"] == q["answerKey"] else ""}">({e(c["label"])}) {e(c["text"])}'
                       f'{" ✓" if c["label"] == q["answerKey"] else ""}</li>' for c in q["question"]["choices"])
        n = len(r["flaws"])
        count = f'<span class="count">{n} to check</span>' if n else '<span class="count none">nothing to check</span>'
        flaws = ""
        for f in r["flaws"]:
            body = (f"<blockquote>{e(f['quote'])}</blockquote>" if f["quote"] else "") + (f"<p>{e(f['explanation'])}</p>" if f["explanation"] else "")
            body += f"<p><b>Try:</b> {e(f['suggestion'])}</p>" if f["suggestion"] else ""
            body += "" if f["checked"] else "<p class=\"why\">Not checked a second time; the first review found it.</p>"
            flaws += f'<div class="flaw"><h3>{e(f["label"])}</h3>{body}<p class="why">Why it matters: {e(f["why"])}</p></div>'
        feats = "".join(f'<div><b>{PARTS[p]}</b>{"".join(f"<span class=chip>{e(x)}</span>" for x in labels)}</div>'
                        for p, labels in r["features"].items() if labels)
        feats = f'<div class="features">{feats}</div>' if feats else ""
        cards.append(f'<article><h2><span>{e(r["id"])}</span>{count}</h2><div class="question"><div>{e(q["question"]["stem"])}</div><ol>{opts}</ol>'
                     f'<div class="feedback">{e(q.get("feedback-text", "") or "")}</div></div>{flaws}{feats}</article>')
    note = (f"{len(report)} questions, {n_flagged} with something to check, most first. Each flag is a common item-writing problem found by a language model, "
            "with the words that show it and one way to revise. The model can be wrong; the flags are for your judgement, not a grade. "
            "The grey tags describe what each question does, without judging it.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{e(title)}</title><style>{CSS}</style></head><body><main><h1>{e(title)}</h1><p class="note">{e(note)}</p>{"".join(cards)}</main></body></html>')
