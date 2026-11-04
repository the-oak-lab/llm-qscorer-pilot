"""Checklist scoring without the network: the prompt, the parser, and the run with a fake engine that answers from the stem."""
import json

import pandas as pd

from qscorer.config import PresenceParams
from qscorer.presence import build_prompt, parse_presence, run_presence, select_questions
from tests.test_baseline import raw

CHECKLIST = [{"id": "f01", "name": "Scenario stem", "definition": "The stem describes a concrete situation.", "criterion": "the stem names an object or event", "part": "stem"},
             {"id": "f02", "name": "Distractor rationale in feedback", "definition": "The feedback says why wrong options are wrong.", "criterion": "Count as present when any wrong option is explained.", "part": "feedback"}]


def question(i, scenario):
    stem = f"A ball rolls down a hill {i}." if scenario else f"What is force {i}?"
    return {"id": f"q{i}", "question": {"stem": stem, "choices": [{"label": c, "text": f"option {c}"} for c in "abcd"]}, "answerKey": "a", "feedback-text": "Because."}


class Engine:
    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            scenario = "rolls" in j.prompt.split("Question:")[1].split("\n")[1]
            k = int(j.id.split("|")[1])
            answer = {"features": [{"id": "f01", "present": scenario}, {"id": "f02", "present": k == 0}]}   # f02 flips between the two calls
            r = raw(j.id, text=json.dumps(answer))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_prompt_and_parser():
    p = build_prompt(question(1, True), CHECKLIST, "Context sentence.")
    assert p.startswith("Context sentence.\n\nBelow is one question") and "Question:\nA ball rolls down a hill 1." in p
    assert "f01. Scenario stem: The stem describes a concrete situation. Count as present when the stem names an object or event." in p
    assert "f02. Distractor rationale in feedback: The feedback says why wrong options are wrong. Count as present when any wrong option is explained." in p
    assert p.index("Checklist:") > p.index("Your task:") > p.index("Feedback:")
    assert parse_presence(json.dumps({"features": [{"id": "f01", "present": True}, {"id": "f02", "present": False}]}), ["f01", "f02"]) == {"f01": True, "f02": False}
    assert parse_presence(json.dumps({"features": [{"id": "f01", "present": True}]}), ["f01", "f02"]) is None
    assert parse_presence(json.dumps({"features": [{"id": "f01", "present": "yes"}, {"id": "f02", "present": False}]}), ["f01", "f02"]) is None


def test_run_presence(tmp_path):
    questions = {q["id"]: q for q in (question(i, i % 2 == 0) for i in range(10))}
    sets = pd.Series({f"q{i}": f"set-{i % 2}" for i in range(10)})
    assert select_questions(questions, ["q0"], 4, sets, 0) and len(select_questions(questions, ["q0"], 4, sets, 0)) == 4
    params = PresenceParams(samples=2)
    result = run_presence(questions, list(questions), CHECKLIST, "Context.", "System.", params, tmp_path, engine=Engine(), progress=lambda s: None)
    assert len(result.samples) == 20 and result.samples.ok.all()
    s = result.scores.set_index("id")
    assert s.f01.tolist() == [1.0 if i % 2 == 0 else 0.0 for i in range(10)] and (s.f02 == 0.5).all() and (s.n_ok == 2).all()
    a = result.summary["agreement"]
    assert a["f01"]["agreement"] == 1.0 and a["f01"]["kappa"] == 1.0 and a["f02"]["agreement"] == 0.0
    assert result.example.startswith("model: ") and (tmp_path / "responses.jsonl").exists()
