"""From DataShop files to the model's observations.

A transaction is kept when it is about an extracted question and falls in a practice phase
or, if the config names one, the baseline phase (a pretest). Transactions are rolled up under
DataShop's rule (one opportunity per presentation, its first attempt is the observation) by
leapfit. Each question becomes its own knowledge component on its practice rows, with
opportunities counted in time order; its baseline rows (first presentation per student only)
form a second component, `<question>|baseline`, seen at one opportunity, so it has an intercept
and no slope. The difference between the two intercepts is the question's jump.

Also here: the practice schedule of each (student, question) pair, attached from the
transactions when the config names a `schedule_column` (the rollup drops it; the schedule is
constant within a pair, and the most frequent value wins where it is not), a question's item
set (the questions seen by the same set of classes) and its observed accuracies, which Stage
0.2 matches on; and the rendering of a question's text for the Gemini stages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from leapfit.data import StepData, from_frame, rollup_transactions

from qscorer.config import DatasetConfig, SlopeParams

KC_MODEL = "Question"
BASELINE = "|baseline"  # suffix of a question's baseline component
SCHEDULE = "Schedule"   # the column of `practice_steps` holding each row's practice schedule


def is_baseline(kc: str) -> bool:
    return kc.endswith(BASELINE)


def load_questions(path: Path) -> dict[str, dict]:
    """The question JSONL as a dict keyed by question id, in file order."""
    with open(path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return {q["id"]: q for q in questions}


def read_datashop(path: Path) -> pd.DataFrame:
    """A DataShop transaction or student-step file, every cell as text, blanks as empty strings."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def practice_steps(tx: pd.DataFrame, config: DatasetConfig, params: SlopeParams,
                   question_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """One row per opportunity, with `KC (Question)`, `Opportunity (Question)` and `Role`
    (`practice` or `baseline`), and a log of what each step kept."""
    log = {"rows": int(len(tx))}
    tx = tx[tx[config.step_column].isin(question_ids)]
    log["rows_about_questions"] = int(len(tx))
    level = tx[config.phases.column]
    practice = level.str.match(config.phases.practice)
    baseline = level.str.match(config.phases.baseline) & ~practice if config.phases.baseline else pd.Series(False, index=tx.index)
    log["rows_in_practice"], log["rows_in_baseline"] = int(practice.sum()), int(baseline.sum())
    tx = tx[practice | baseline]
    if params.timeouts == "drop" and config.timed_out is not None:
        m = config.timed_out
        no_answer = tx[m.column].eq(m.value) & tx[m.input_column].eq("")
        log["no_answer_timeouts_dropped"] = int(no_answer.sum())
        tx = tx[~no_answer]

    if "First Attempt" in tx.columns:  # already a student-step table
        steps = tx.copy()
        log["input"] = "student-step"
    else:
        steps = rollup_transactions(tx)
        log["input"] = "transactions"
    log["opportunities"] = int(len(steps))

    steps = steps.copy()
    if config.schedule_column is not None:
        pair = [config.student_column, config.step_column]
        counts = tx.groupby(pair + [config.schedule_column]).size().reset_index(name="n")
        counts = counts.sort_values(["n", config.schedule_column], ascending=[False, True], kind="stable").drop_duplicates(pair)
        key = counts.set_index(pair)[config.schedule_column]
        steps[SCHEDULE] = np.asarray(pd.MultiIndex.from_frame(steps[pair]).map(key))
        log["schedules"] = steps[SCHEDULE].value_counts().to_dict()
    steps["Role"] = np.where(steps[config.phases.column].str.match(config.phases.practice), "practice", "baseline")
    view = pd.to_numeric(steps.get("Problem View", pd.Series("", index=steps.index)), errors="coerce")
    order = pd.DataFrame({"student": steps[config.student_column], "time": steps["First Transaction Time"],
                          "view": view, "row": np.arange(len(steps))}).sort_values(["student", "time", "view", "row"])
    steps = steps.loc[order.index]
    within = steps.groupby([config.student_column, config.step_column, "Role"]).cumcount() + 1
    base = steps["Role"].eq("baseline")
    repeats = base & (within > 1)  # a repeated pretest: only the first presentation is the baseline
    log["baseline_repeats_dropped"] = int(repeats.sum())
    steps, within = steps[~repeats], within[~repeats]
    steps[f"KC ({KC_MODEL})"] = np.where(steps["Role"].eq("baseline"), steps[config.step_column] + BASELINE, steps[config.step_column])
    steps[f"Opportunity ({KC_MODEL})"] = within.astype(str)
    if params.max_opportunities is not None:
        keep = steps["Role"].eq("baseline") | (within <= params.max_opportunities)
        steps = steps[keep]
        log["opportunities_within_cap"] = int(len(steps))
    log["students"] = int(steps[config.student_column].nunique())
    log["questions_with_data"] = int(steps[config.step_column].nunique())
    return steps.reset_index(drop=True), log


def to_step_data(steps: pd.DataFrame) -> StepData:
    """leapfit's observations for the `Question` KC model."""
    return from_frame(steps, KC_MODEL)


def item_sets(tx: pd.DataFrame, config: DatasetConfig, ids: list[str]) -> dict[str, str]:
    """An item-set label per question: questions seen by the same set of `set_column` values
    (e.g. the same classes) share a set. Labels are `set-1`, `set-2`, ... in order of first
    appearance. Without a `set_column` every question is in `set-1`."""
    if config.set_column is None:
        return {q: "set-1" for q in ids}
    t = tx[tx[config.step_column].isin(ids)]
    signature = t.groupby(config.step_column)[config.set_column].agg(frozenset)
    labels: dict[frozenset, str] = {}
    out = {}
    for q in ids:
        sig = signature.get(q, frozenset())
        labels.setdefault(sig, f"set-{len(labels) + 1}")
        out[q] = labels[sig]
    return out


def observed_accuracy(steps: pd.DataFrame, config: DatasetConfig, ids: list[str]) -> pd.DataFrame:
    """Per question, the observed share correct at the baseline and on the first practice
    attempt, with the number of students behind each."""
    correct = steps["First Attempt"].eq("correct").astype(float)
    opp = steps[f"Opportunity ({KC_MODEL})"].astype(int)
    out = pd.DataFrame(index=pd.Index(ids, name="id"))
    for name, mask in [("pretest", steps.Role.eq("baseline")), ("first_practice", steps.Role.eq("practice") & opp.eq(1))]:
        g = correct[mask].groupby(steps.loc[mask, config.step_column])
        out[f"{name}_accuracy"] = g.mean()
        out[f"n_{name}"] = g.size()
    return out


BULLET = re.compile(r"\s(?:-|\u2022)\s+(?=[A-Z$(*\u201c\"'])")  # a bullet marker: " - " or " \u2022 " before a capital, formula, bold or quote


def render_feedback(text: str) -> str:
    """The feedback as bullet lines when the export wrote it as bullets, else as it is. The
    exports flattened bullet lists onto one line: "- A ... - B ..." (exp4, exp3) or "\u2022 A ... \u2022 B"
    (exp3). A " - " before a lowercase word is a dash inside a bullet, not a new one, and a
    minus sign inside $...$ math is never a marker."""
    text = text.strip()
    if not text.startswith(("- ", "\u2022")):
        return text
    text = text.lstrip("-\u2022").lstrip()  # the first marker
    items, current = [], ""
    for i, segment in enumerate(text.split("$")):  # odd segments are inside $...$
        if i % 2:
            current += "$" + segment + "$"
            continue
        pieces = BULLET.split(segment)
        current += pieces[0]
        for piece in pieces[1:]:
            items.append(current)
            current = piece
    items.append(current)
    return "\n".join(f"- {t.strip()}" for t in items if t.strip())


def render_question(q: dict, header: str = "Question") -> str:
    """The question as the Gemini stages show it: `header`, stem, options, key, feedback."""
    lines = [f"{header}:", q["question"]["stem"]]
    lines += [f"({c['label']}) {c['text']}" for c in q["question"]["choices"]]
    lines += ["", f"Correct answer: ({q['answerKey']})", "", "Feedback:", render_feedback(q.get("feedback-text", "") or "")]
    return "\n".join(lines)
