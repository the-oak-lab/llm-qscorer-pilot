"""The claimed direction of each checklist item without the network: the prompt batches, the parser, and the claim from the asked polarity and the relation."""
import json

import pandas as pd

from qscorer.config import CanonParams
from qscorer.direction import build_jobs, parse_relations, run_direction
from tests.test_baseline import raw

CHECKLIST = [{"id": "f01", "name": "Refutational feedback", "definition": "The feedback says why wrong options are wrong.", "criterion": "Count as present when a wrong option is explained.", "part": "feedback"},
             {"id": "f02", "name": "Longer key", "definition": "The correct option is longer.", "criterion": "Count as present when the key is the longest.", "part": "options"}]
# f01: "Explicit refutation" proposed as good (same -> good), "No distractor rationale" proposed as bad (opposite -> good), "Long stem" (neither)
# f02: "Long correct option" proposed as bad (same -> bad), three times
FEATURES = pd.DataFrame([("E5", "Explicit refutation", "f01"), ("E6", "No distractor rationale", "f01"), ("E6", "Long stem", "f01")] +
                        [("E2", "Long correct option", "f02")] * 3, columns=["condition", "name", "item"]).assign(
    batch="b", sample=0, feature=range(6), definition="d")
RELATION = {"Explicit refutation": "same", "No distractor rationale": "opposite", "Long stem": "neither", "Long correct option": "same"}


class Engine:
    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            lines = j.prompt.split("Features:\n")[1].split("\n\n")[0].splitlines()
            answer = {"features": [{"index": k + 1, "relation": RELATION[line.split(". ", 1)[1].split(":")[0]]} for k, line in enumerate(lines)]}
            r = raw(j.id, text=json.dumps(answer))
            if accept(r):
                done[j.id] = r
        return done


def test_jobs_and_parser():
    jobs, rows = build_jobs(FEATURES, CHECKLIST, CanonParams(), "System.", batch=2)
    assert [j.id for j in jobs] == ["f01|0", "f01|2", "f02|0", "f02|2"] and rows["f01|2"] == [2] and rows["f02|2"] == [5]
    assert "Checklist item: Refutational feedback. The feedback says why wrong options are wrong." in jobs[0].prompt and "1. Explicit refutation: d" in jobs[0].prompt
    assert parse_relations(json.dumps({"features": [{"index": 1, "relation": "same"}, {"index": 2, "relation": "opposite"}]}), 2) == {1: "same", 2: "opposite"}
    assert parse_relations(json.dumps({"features": [{"index": 1, "relation": "same"}]}), 2) is None
    assert parse_relations(json.dumps({"features": [{"index": 1, "relation": "absent"}]}), 1) is None


def test_run_direction(tmp_path):
    rows, items, summary = run_direction(FEATURES, CHECKLIST, CanonParams(), "System.", tmp_path, batch=2, engine=Engine(), progress=lambda s: None)
    assert rows.asked.tolist() == ["good", "bad", "bad", "bad", "bad", "bad"]
    assert rows.claim.fillna("").tolist() == ["good", "good", "", "bad", "bad", "bad"]
    t = items.set_index("item")
    assert t.claimed_good_share["f01"] == 1.0 and t.claimed_good_share["f02"] == 0.0 and t.neither["f01"] == 1 and t.unjudged.sum() == 0
    assert summary["answered"] == summary["calls"] == 4
