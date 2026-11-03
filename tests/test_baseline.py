"""Stage 1 without the network: prompts, parsing, tables, consistency, evaluation, and the run with a fake engine."""
import json

import numpy as np
import pandas as pd
import pytest

from qscorer.baseline import (accept, build_jobs, build_prompt, consistency, evaluate, parse_answer, rating_table, render_feedback,
                              run_baseline, sample_table, select_questions)
from qscorer.config import BaselineParams, DatasetConfig, PhaseFilter
from qscorer.gemini import Job, parse

CONFIG = DatasetConfig(name="fake", transactions="x.txt", questions="x.jsonl", phases=PhaseFilter(practice="^learning$"), exclude=["q3"])
Q = {f"q{i}": {"id": f"q{i}", "question": {"stem": f"What is {i} + {i}?", "choices": [{"label": "a", "text": str(2 * i)}, {"label": "b", "text": "0"},
                                                                                     {"label": "c", "text": "1"}, {"label": "d", "text": "x"}]},
               "answerKey": "a", "feedback-text": f"- {i} and {i} make {2 * i}."} for i in range(1, 7)}


def raw(rsp_id, answer=None, text=None, error=None, thought="thinking...", finish="STOP"):
    if error:
        return {"id": rsp_id, "error": error}
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text if text is not None else json.dumps({"reason": "because", "answer": answer})}]
    return {"id": rsp_id, "candidates": [{"content": {"parts": parts}, "finish_reason": finish}],
            "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 20, "thoughts_token_count": 50}}


class FakeEngine:
    """Answers improve with the question number and known against it; q5 fails once, then works."""
    def __init__(self):
        self.calls = 0

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for attempt in range(passes):
            for j in jobs:
                if j.id in done:
                    continue
                self.calls += 1
                qid, task, k = j.id.split("|")
                n = int(qid[1:])
                if qid == "q5" and attempt == 0:
                    r = raw(j.id, error="503 overloaded")
                else:
                    r = raw(j.id, answer=min(100, 10 * n + int(k)) if task == "improve" else max(0, 90 - 10 * n - int(k)))
                with open(log, "a") as f:
                    f.write(json.dumps(r) + "\n")
                if accept(r):
                    done[j.id] = r
        return done


def test_prompt_and_jobs():
    p = build_prompt(Q["q1"], "known", CONFIG)
    assert CONFIG.prompt.context in p and "Question:\nWhat is 1 + 1?" in p and "(a) 2" in p and "\n\nCorrect answer: (a)\n\nFeedback:\n- 1 and 1 make 2." in p and "- 1 and 1 make 2.\n\n\nYour task:\nGiven the features of the question and its feedback, estimate how many students (out of 100) already know" in p and '"answer"' in p
    params = BaselineParams(samples=2, prompts=["improve", "known"], limit=None)
    ids = select_questions(Q, CONFIG, params, None)
    assert ids == ["q1", "q2", "q4", "q5", "q6"]  # q3 excluded
    jobs = build_jobs(Q, ids, CONFIG, params)
    assert len(jobs) == 5 * 2 * 2 and jobs[0].id == "q1|improve|0" and jobs[1].config["seed"] == 1
    assert jobs[0].config["thinking_config"] == {"thinking_level": "MEDIUM", "include_thoughts": True} and jobs[0].config["response_mime_type"] == "application/json"
    sets = pd.Series({"q1": "s1", "q2": "s1", "q4": "s2", "q5": "s2", "q6": "s2"})
    few = select_questions(Q, CONFIG, BaselineParams(limit=2), sets)
    assert len(few) == 2 and sets[few[0]] == "s1" and sets[few[1]] == "s2"  # spread over sets


def test_render_feedback():
    assert render_feedback("- The Sun moves - Earth rotates - The Sun is stationary relative to Earth - it's Earth that moves") == \
        "- The Sun moves\n- Earth rotates\n- The Sun is stationary relative to Earth - it's Earth that moves"
    assert render_feedback("\u2022 Non-metals are brittle \u2022 They have a dull appearance") == "- Non-metals are brittle\n- They have a dull appearance"
    assert render_feedback("- $q = mc\\Delta T = (23.0 - 20.0)$ joules - The answer is 1,200 J - Then 5 - 3 = 2") == "- $q = mc\\Delta T = (23.0 - 20.0)$ joules\n- The answer is 1,200 J\n- Then 5 - 3 = 2"
    prose = "A neutral iron atom has 26 electrons: $26 - 3 = 23$ electrons - that is the count."
    assert render_feedback(prose) == prose  # not a bullet list: left alone


def test_parse():
    r = parse(raw("q1|improve|0", answer=42))
    assert r.ok and r.thought == "thinking..." and r.thought_tokens == 50 and parse_answer(r.text) == (42, "because")
    assert not parse(raw("x", error="boom")).ok
    assert parse_answer('{"reason": "r", "answer": 140}') == (None, "r") and parse_answer("not json") == (None, None)
    assert parse_answer('{"reason": "r", "answer": 55.6}') == (56, "r")
    assert accept(raw("x", answer=3)) and not accept(raw("x", text="garbage")) and not accept(raw("x", text=None, thought="t", finish="MAX_TOKENS"))


def test_tables_consistency_and_evaluation():
    params = BaselineParams(samples=3, prompts=["improve", "known"])
    jobs = build_jobs(Q, list(Q), CONFIG, params)
    records = {j.id: raw(j.id, answer=(10 * int(j.id[1]) + int(j.id[-1])) if "improve" in j.id else 90 - 10 * int(j.id[1])) for j in jobs if j.id != "q6|known|2"}
    samples = sample_table(records, jobs)
    assert len(samples) == 36 and samples.error.eq("not asked").sum() == 1 and samples.loc[samples.id.eq("q1") & samples.prompt.eq("known"), "rating"].tolist() == [20, 20, 20]
    ratings = rating_table(samples)
    assert ratings[ratings.prompt == "known"].n_ok.tolist() == [3, 3, 3, 3, 3, 2] and abs(ratings[(ratings.id == "q1") & (ratings.prompt == "improve")].answer.iloc[0] - 11) < 1e-9
    c = consistency(samples)
    assert c["known"]["icc_single"] > 0.99 and c["known"]["samples_for_0.90"] == 1 and 0.9 < c["improve"]["icc_single"] < 1
    slopes = pd.DataFrame({"id": list(Q), "set": ["s1"] * 3 + ["s2"] * 3, "slope": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "first_practice_accuracy": [.5, .6, .4, .5, .7, .3]})
    pairs_q = pd.DataFrame({"id": list(Q), "pool": ["low", "", "high", "low", "", "high"], "partner": ["q3", "", "q1", "q6", "", "q4"], "selected": [True, False, True, True, False, True]})
    e = evaluate(ratings, slopes, pairs_q, boot=50)
    assert abs(e["improve"]["spearman"] - 1) < 1e-9 and e["improve"]["auc_high_vs_low"] == 0.75 and e["improve"]["high_member_rated_higher"] == 1.0  # q3 (31) < q4 (41) across pairs, but each pair is ordered and e["improve"]["kept_pairs_with_both_rated"] == 2
    assert abs(e["known"]["spearman"] - 1) < 1e-9 and "within_sets" in e["known"]
    assert evaluate(ratings, None, None)["note"].startswith("no Stage 0")


def test_run_with_a_fake_engine(tmp_path):
    with open(tmp_path / "q.jsonl", "w") as f:
        for q in Q.values():
            f.write(json.dumps(q) + "\n")
    config = CONFIG.model_copy(update={"questions": tmp_path / "q.jsonl"})
    params = BaselineParams(samples=2, limit=4, passes=2)
    engine = FakeEngine()
    result = run_baseline(config, params, tmp_path / "out", engine=engine, progress=lambda s: None)
    assert result.summary["questions_asked"] == 4 and result.summary["calls"] == 16 and result.summary["answered"] == 16
    assert engine.calls == 16 + 4 * (1 if "q5" in set(result.samples.id) else 0)  # q5's four calls were resent once
    assert (tmp_path / "out" / "responses.jsonl").exists() and "=== known ===" in result.example
    assert result.evaluation["note"].startswith("no Stage 0") and result.summary["tokens"]["thought_tokens"] == 16 * 50
    again = run_baseline(config, params, tmp_path / "out", engine=engine, progress=lambda s: None)  # resumes from the log: no new calls
    assert again.summary["answered"] == 16
