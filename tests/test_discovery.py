"""Stage 2.2 without the network: prompts per condition, jobs, parsing, the evidence check, the tables and the run with a fake engine."""
import json

import pandas as pd

from qscorer.batches import BatchParams, plan_batches
from qscorer.config import DatasetConfig, DiscoveryParams, PhaseFilter
from qscorer.discovery import accept, build_jobs, build_prompt, evidence_found, normalise, parse_features, run_discovery, select_batches, shown_label
from qscorer.gemini import parse
from tests.test_baseline import FakeEngine, raw

CONFIG = DatasetConfig(name="fake", transactions="x.txt", questions="x.jsonl", phases=PhaseFilter(practice="^learning$"))
Q = {f"{p}{i}": {"id": f"{p}{i}", "question": {"stem": f"What is {i} + {i}, {p}?", "choices": [{"label": "a", "text": str(2 * i)}, {"label": "b", "text": "0"},
                                                                                              {"label": "c", "text": "1"}, {"label": "d", "text": "x"}]},
                 "answerKey": "a", "feedback-text": f"- {i} and {i} make {2 * i}. - Adding is counting on."} for p in "hl" for i in range(8)}
PAIRS = pd.DataFrame({"set": ["s1"] * 8, "high": [f"h{i}" for i in range(8)], "low": [f"l{i}" for i in range(8)],
                      "accuracy_high": [0.5] * 8, "accuracy_low": [0.5] * 8})
PLAN, _ = plan_batches(PAIRS, BatchParams(rounds=2, seed=0))


def reply(rsp_id, features):
    return raw(rsp_id, text=json.dumps({"features": features}))


class Engine:
    """Names one feature in every call, attributed to questions 1 and 2 with a real quote and a made-up one; E3 calls fail once."""
    def __init__(self):
        self.calls = 0

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for attempt in range(passes):
            for j in jobs:
                if j.id in done:
                    continue
                self.calls += 1
                if j.id.split("|")[1] == "E3" and attempt == 0:
                    r = raw(j.id, error="503")
                else:
                    r = reply(j.id, [{"name": "Counting feedback", "definition": "Feedback restates addition as counting.", "mechanism": "Gives a strategy.",
                                      "questions": [{"question": 1, "exhibits": True, "evidence": "Adding is counting on."},
                                                    {"question": 2, "exhibits": False, "evidence": "not in the text"}]}])
                with open(log, "a") as f:
                    f.write(json.dumps(r) + "\n")
                if accept(r):
                    done[j.id] = r
        return done


def test_prompts_and_jobs():
    mixed = PLAN[PLAN.batch == "r1-s1-mixed-01"]
    p7, p9, p5 = (build_prompt(mixed, Q, c, CONFIG) for c in ("E7", "E9", "E5"))
    assert "Question 1" in p5 and "Question 4:" in p5 and "reported label" not in p5 and "facilitate improvement" in p5 and '"features"' in p5
    assert "HIGH-LEARNING questions show greater improvement" in p7 and "both groups" in p7
    first = mixed[mixed.position == 1].pool.iloc[0]
    assert f"Question 1 (reported label: {shown_label(first, 'true')}):" in p7 and f"Question 1 (reported label: {shown_label(first, 'inverted')}):" in p9
    assert shown_label("high", "inverted") == "LOW-LEARNING" and shown_label("low", None) is None
    assert p7.replace("HIGH-LEARNING", "X").replace("LOW-LEARNING", "HIGH-LEARNING").replace("X", "LOW-LEARNING").split("\n\n\n")[0] == p9.split("\n\n\n")[0]  # labels swapped, nothing else
    good = PLAN[PLAN.batch == "r1-s1-good-01"]
    assert "impede improvement" in build_prompt(good, Q, "E2", CONFIG) and "LOW-LEARNING questions show less improvement" in build_prompt(mixed, Q, "E8", CONFIG)
    jobs = build_jobs(PLAN, Q, CONFIG, DiscoveryParams(samples=2))
    assert len(jobs) == 2 * 2 * (4 * 6 + 2 * 2 + 2 * 2)   # 2 rounds x 2 samples x (4 mixed batches x 6 conditions + 2 good x 2 + 2 bad x 2)
    assert jobs[0].id.endswith("|0") and jobs[1].config["seed"] == 1 and jobs[0].config["max_output_tokens"] == 24576
    few = select_batches(PLAN, DiscoveryParams(limit=1))
    assert few["round"].eq(1).all() and few.batch.nunique() == 3 and set(few.kind) == {"mixed", "good", "bad"}


def test_parse_and_accept():
    f = parse_features(json.dumps({"features": [{"name": "n", "definition": "d", "mechanism": "m",
                                                 "questions": [{"question": 2, "evidence": "e"}, {"question": 3, "exhibits": False, "evidence": "c"}, {"question": 7, "evidence": "x"}]}]}))
    assert f == [{"name": "n", "definition": "d", "mechanism": "m", "questions": [{"question": 2, "exhibits": True, "evidence": "e"}, {"question": 3, "exhibits": False, "evidence": "c"}]}]
    assert parse_features(json.dumps({"features": []})) == [] and parse_features("nope") is None and parse_features(json.dumps({"features": "x"})) is None
    assert accept(reply("x", [])) and not accept(raw("x", text="garbage")) and not accept(raw("x", error="boom"))
    assert parse(reply("x", [])).ok


def test_evidence_check():
    from qscorer.data import render_question
    text = normalise(render_question(Q["h1"], "Question 2 (reported label: HIGH-LEARNING)"))
    assert text.startswith("what is 1 + 1, h?") and "- 1 and 1" not in text and "question 2" not in text
    assert evidence_found("1 and 1 make 2.", text) and evidence_found("1 AND 1  make 2. Adding is counting on.", text)   # across bullets
    assert evidence_found("Stem: What is 1 + 1, h? Feedback: Adding is counting on.", text)                           # stitched passages
    assert not evidence_found("1 and 1 make 3.", text) and not evidence_found("", text) and not evidence_found("Stem: nothing here at all.", text)
    assert evidence_found("The feedback says '1 and 1 make 2.' but never mentions subtraction", text)               # quoted segment inside prose
    assert not evidence_found("The feedback says '1 and 1 make 3.' but never mentions subtraction", text)


def test_run_with_a_fake_engine(tmp_path):
    with open(tmp_path / "q.jsonl", "w") as f:
        for q in Q.values():
            f.write(json.dumps(q) + "\n")
    config = CONFIG.model_copy(update={"questions": tmp_path / "q.jsonl"})
    params = DiscoveryParams(samples=2, limit=1, passes=2)
    engine = Engine()
    result = run_discovery(config, params, PLAN, tmp_path / "out", engine=engine, progress=lambda s: None)
    s = result.summary
    assert s["batches_asked"] == 3 and s["calls"] == 2 * (6 + 2 + 2) and s["answered"] == 20 and s["failed"] == 0
    assert engine.calls == 20 + 2                                     # the two E3 calls were resent once
    assert len(result.features) == 20 and (result.features.n_questions == 1).all() and (result.features.evidence_found == 0.5).all()  # question 2 is a contrast
    assert len(result.attributions) == 40 and result.attributions.evidence_found.mean() == 0.5 and result.attributions.exhibits.mean() == 0.5
    assert s["per_condition"]["E5"]["contrasts_per_feature"] == 1.0
    a = result.attributions
    assert set(a[a.condition == "E7"].label.dropna()) <= {"HIGH-LEARNING", "LOW-LEARNING"} and a[a.condition == "E5"].label.isna().all()
    e7, e9 = a[a.condition == "E7"], a[a.condition == "E9"]
    assert (e7.pool.to_numpy() == e9.pool.to_numpy()).all() and (e7.label.to_numpy() != e9.label.to_numpy()).all()   # same questions, opposite labels
    assert s["per_condition"]["E7"]["attributions_to_high_members"] + s["per_condition"]["E9"]["attributions_to_high_members"] == 2 * s["per_condition"]["E5"]["attributions_to_high_members"]
    assert "=== r1-s1-good-01|E1|0 ===" in result.example and (tmp_path / "out" / "responses.jsonl").exists()
