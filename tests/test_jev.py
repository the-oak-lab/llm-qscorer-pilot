"""Jev scoring without the network: the questions asked, the request body, the answer parser, and a run with a fake engine."""
import json

import pytest

from qscorer.jev import (
    build_jobs,
    load_rewordings,
    model_id,
    nouls,
    questions_for,
    run_jev,
    statement,
)
from qscorer.llm import ROOT

CHECKLIST = [{"id": "f01", "name": "Scenario stem", "definition": "The stem describes a concrete situation.", "criterion": "the stem names an object or event", "part": "stem"},
             {"id": "f02", "name": "Distractor rationale", "definition": "The feedback says why wrong options are wrong.", "criterion": "Count as present when any wrong option is explained.", "part": "feedback"}]
Q = {"q1": {"id": "q1", "question": {"stem": "A ball rolls.", "choices": [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}]}, "answerKey": "b", "feedback-text": "Because."},
     "q2": {"id": "q2", "question": {"stem": "What is force?", "choices": [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}]}, "answerKey": "a"}}


def test_statement_and_questions():
    assert statement("Count as present when the question stem includes data.") == "The question stem includes data."
    assert statement("count as present when arriving at x requires y") == "Arriving at x requires y."
    qs = questions_for(CHECKLIST, {"f02": "A wrong option is explained."})
    assert qs["f01"] == {"type": "noul", "instructions": "The stem names an object or event.",
                         "criteria": {"true": "Scenario stem: The stem describes a concrete situation.", "false": "The question does not show this feature (Scenario stem)."}}
    assert qs["f02"] == {"type": "noul", "instructions": "A wrong option is explained."}
    with pytest.raises(ValueError):
        questions_for(CHECKLIST, {"f09": "Not an item."})


def test_body():
    body = json.loads(build_jobs(Q, ["q1"], CHECKLIST, "Context.", {}, "jev-1.14.0")[0].prompt)
    assert body["model"] == "jev-1.14.0" and list(body["questions"]) == ["f01", "f02"]
    assert body["state"] == {"context": "Context.", "question": "A ball rolls.", "options": {"a": "x", "b": "y"}, "correct_answer": "b", "feedback": "Because."}


def test_model_id():
    assert model_id("jev-1.14.0", "direct") == "jev-1.14.0" and model_id("jev-1.14.0", "openrouter") == "typesafe/jev-1.14"
    assert model_id("typesafe/jev-1.14", "direct") == "jev-1.14.0"
    with pytest.raises(ValueError):
        model_id("jev-9.0.0", "openrouter")


def test_nouls():
    ok = {"answers": {"f01": {"type": "noul", "noul": 0.9}, "f02": {"type": "noul", "noul": 0}}}
    assert nouls(ok, ["f01", "f02"]) == {"f01": 0.9, "f02": 0.0}
    assert nouls({"answers": {"f01": {"type": "noul", "noul": 0.9}, "f02": {"type": "refusal"}}}, ["f01", "f02"]) is None
    assert nouls({"answers": {"f01": {"type": "noul", "noul": 0.9}}}, ["f01", "f02"]) is None
    assert nouls({"error": "HTTP 500"}, ["f01"]) is None


class Engine:
    model = "jev-1.14.0"

    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            r = {"id": j.id, "model": "jev-1.14.0", "usage": {"input_tokens": 500, "output_tokens": 40},
                 "answers": {"f01": {"type": "noul", "noul": 0.8}, "f02": {"type": "noul", "noul": 0.2}}} if j.id.startswith("q1") else {"id": j.id, "error": "HTTP 529"}
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_run(tmp_path):
    result = run_jev(Q, list(Q), CHECKLIST, "Context.", {}, tmp_path, engine=Engine(), progress=lambda s: None)
    assert result.scores.to_dict("records") == [{"id": "q1", "n_ok": 1, "f01": 0.8, "f02": 0.2}]
    s = result.summary
    assert s["answered"] == 1 and s["failed"] == 1 and s["tokens"]["prompt_tokens"] == 500 and s["served_by"] == {"jev-1.14.0": 1}
    assert s["params"]["engine"] == "jev" and s["params"]["model"] == "jev-1.14.0" and s["params"]["present"] == 0.7
    assert result.samples.set_index("id").loc["q2", "error"] == "HTTP 529"


def test_shipped_model_matches_config():
    """The shipped Jev model asks the rewordings in configs/jev-rewordings.json, and every item gets a question."""
    model = json.load(open(ROOT / "models" / "spacing-exp4-jev" / "model.json"))
    assert model["scoring"]["engine"] == "jev" and model["scoring"]["rewordings"] == load_rewordings()
    assert len(questions_for(model["items"], model["scoring"]["rewordings"])) == len(model["items"]) == 45
    assert model["params"]["flaw_present"] == 0.7
    gemini = json.load(open(ROOT / "models" / "spacing-exp4" / "model.json"))
    assert model["items"] == gemini["items"] and model["flaws"] == gemini["flaws"]   # same checklist, same list of flaws
