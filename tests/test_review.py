"""Teacher review without the network: candidates from the scores, the prompt, the parser, the quote check, the run with a fake engine, and the report."""
import json

import pandas as pd

from qscorer.config import PresenceParams
from qscorer.review import build_prompt, build_report, candidates, parse_review, quote_found, render_html, run_review
from tests.test_baseline import raw

CHECKLIST = [{"id": "f01", "name": "Long key", "definition": "The correct option is longer.", "criterion": "the key is the longest option", "part": "options"},
             {"id": "f02", "name": "Scenario stem", "definition": "The stem describes a situation.", "criterion": "the stem names an event", "part": "stem"},
             {"id": "f03", "name": "Factual error", "definition": "Something is wrong.", "criterion": "a claim is false", "part": "feedback"},
             {"id": "f04", "name": "Overlap", "definition": "Shares words with other items.", "criterion": "another item is cued", "part": "cross-item"}]
GROUPS = {"f01": {"group": "flaw", "label": "The correct option is longer", "why": "Students learn the cue."},
          "f02": {"group": "feature", "label": "Sets the question in a scenario"},
          "f03": {"group": "flaw", "label": "Possible factual error", "why": "Students remember it."},
          "f04": {"group": "hidden", "label": "Overlap"}}


def question(i):
    return {"id": f"q{i}", "question": {"stem": f"A ball rolls down hill {i}. What happens?",
                                         "choices": [{"label": "a", "text": "It speeds up because gravity pulls it along the slope"},
                                                     {"label": "b", "text": "It stops"}, {"label": "c", "text": "It floats"}, {"label": "d", "text": "It shrinks"}]},
            "answerKey": "a", "feedback-text": "Gravity has a component along the slope."}


class Engine:
    """Confirms f01 with a quote from option (a), rejects f03; question q2's call never parses."""
    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            ids = [line.split(".")[0] for line in j.prompt.split("Possible problems:\n")[1].split("\n\n")[0].splitlines()]
            flags = [{"id": i, "applies": i == "f01", "quote": "It speeds up because gravity ... along the slope" if i == "f01" else "",
                      "explanation": "Option (a) is the longest." if i == "f01" else "No error.", "suggestion": "Shorten (a)." if i == "f01" else ""} for i in ids]
            r = raw(j.id, text="not json" if j.id == "q2" else json.dumps({"flags": flags}))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_prompt_parser_and_quote():
    p = build_prompt(question(1), CHECKLIST[:1], "Context sentence.")
    assert p.startswith("Context sentence.") and "Possible problems:\nf01. Long key: The correct option is longer. Count as present when the key is the longest option." in p
    good = {"flags": [{"id": "f01", "applies": True, "quote": " q ", "explanation": "e", "suggestion": "s"}]}
    assert parse_review(json.dumps(good), ["f01"]) == {"f01": {"applies": True, "quote": "q", "explanation": "e", "suggestion": "s"}}
    assert parse_review(json.dumps(good), ["f01", "f03"]) is None and parse_review("{", ["f01"]) is None
    assert quote_found("“it  SPEEDS up because gravity … along the slope”", question(1))
    assert not quote_found("it slows down", question(1)) and not quote_found("...", question(1))


def test_run_review_and_report(tmp_path):
    questions = {q["id"]: q for q in (question(i) for i in range(4))}
    scores = pd.DataFrame({"id": list(questions), "f01": [1.0, 1.0, 1.0, 0.0], "f02": [1.0, 0.0, 1.0, 0.5], "f03": [0.5, 0.0, 0.0, 0.0], "f04": [1.0] * 4})
    assert candidates(scores, GROUPS, 0.5) == {"q0": ["f01", "f03"], "q1": ["f01"], "q2": ["f01"], "q3": []}
    flags = run_review(questions, scores, CHECKLIST, GROUPS, "Context.", "System.", PresenceParams(), tmp_path / "log.jsonl", engine=Engine(), progress=lambda s: None)
    assert len(flags) == 4 and flags.checked.tolist() == [True, True, True, False]          # q0 f01, q0 f03, q1 f01, q2 f01 in row order
    f = flags.set_index(["id", "item"])
    assert f.loc[("q0", "f01"), "applies"] and f.loc[("q0", "f01"), "quote_found"] and not f.loc[("q0", "f03"), "applies"]
    assert not f.loc[("q2", "f01"), "checked"]

    report = build_report(questions, scores, flags, CHECKLIST, GROUPS, 0.5)
    by_id = {r["id"]: r for r in report}
    assert [r["id"] for r in report] == ["q0", "q1", "q2", "q3"]
    assert [x["id"] for x in by_id["q0"]["flaws"]] == ["f01"]                                  # f03 rejected by the check
    assert by_id["q2"]["flaws"][0]["checked"] is False                                       # the check failed: shown, marked unchecked
    assert by_id["q3"]["flaws"] == [] and by_id["q3"]["features"]["stem"] == ["Sets the question in a scenario"]
    assert by_id["q1"]["features"] == {"stem": [], "options": [], "feedback": []}            # hidden f04 never shown
    page = render_html(report, questions, "Review")
    assert page.count("<article>") == 4 and "Shorten (a)." in page and "Not checked a second time" in page and "Overlap" not in page
