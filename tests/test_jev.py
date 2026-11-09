"""Jev scoring without the network: the request body, the answer parser, and a run with a fake engine."""
import json

from qscorer.jev import build_jobs, nouls, run_jev, statement

CHECKLIST = [{"id": "f01", "name": "Scenario stem", "definition": "The stem describes a concrete situation.", "criterion": "the stem names an object or event", "part": "stem"},
             {"id": "f02", "name": "Distractor rationale", "definition": "The feedback says why wrong options are wrong.", "criterion": "any wrong option is explained", "part": "feedback"}]
Q = {"q1": {"id": "q1", "question": {"stem": "A ball rolls.", "choices": [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}]}, "answerKey": "b", "feedback-text": "Because."},
     "q2": {"id": "q2", "question": {"stem": "What is force?", "choices": [{"label": "a", "text": "x"}, {"label": "b", "text": "y"}]}, "answerKey": "a"}}


def test_body():
    body = json.loads(build_jobs(Q, ["q1"], CHECKLIST, "Context.", "typesafe/jev-1.14")[0].prompt)
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"] == {"context": "Context.", "question": "A ball rolls.", "options": {"a": "x", "b": "y"}, "correct_answer": "b", "feedback": "Because."}
    assert body["questions"]["f01"] == {"type": "noul", "instructions": "The stem names an object or event."}
    defn = json.loads(build_jobs(Q, ["q1"], CHECKLIST, "Context.", "typesafe/jev-1.14", "statement+definition")[0].prompt)["questions"]["f01"]
    assert defn["instructions"] == "The stem names an object or event."
    assert defn["criteria"] == {"true": "Scenario stem: The stem describes a concrete situation.", "false": "The question does not show this feature (Scenario stem)."}


def test_statement():
    assert statement("Count as present when the question stem includes data.") == "The question stem includes data."
    assert statement("count as present when arriving at x requires y") == "Arriving at x requires y."


def test_nouls():
    ok = {"answers": {"f01": {"type": "noul", "noul": 0.9}, "f02": {"type": "noul", "noul": 0}}}
    assert nouls(ok, ["f01", "f02"]) == {"f01": 0.9, "f02": 0.0}
    assert nouls({"answers": {"f01": {"type": "noul", "noul": 0.9}, "f02": {"type": "refusal"}}}, ["f01", "f02"]) is None
    assert nouls({"answers": {"f01": {"type": "noul", "noul": 0.9}}}, ["f01", "f02"]) is None
    assert nouls({"error": "HTTP 500"}, ["f01"]) is None


class Engine:
    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            r = {"id": j.id, "usage": {"input_tokens": 500, "output_tokens": 40},
                 "answers": {"f01": {"type": "noul", "noul": 0.8}, "f02": {"type": "noul", "noul": 0.2}}} if j.id.startswith("q1") else {"id": j.id, "error": "HTTP 529"}
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


def test_run(tmp_path):
    result = run_jev(Q, list(Q), CHECKLIST, "Context.", tmp_path, engine=Engine(), progress=lambda s: None)
    assert result.scores.to_dict("records") == [{"id": "q1", "n_ok": 1, "f01": 0.8, "f02": 0.2}]
    assert result.summary["answered"] == 1 and result.summary["failed"] == 1 and result.summary["tokens"]["prompt_tokens"] == 500
    assert result.samples.set_index("id").loc["q2", "error"] == "HTTP 529"


def test_split():
    from qscorer.jev import combine, questions_for
    splits = {"f02": ["Wrong options are explained.", "A common error is warned against."]}
    qs = questions_for(CHECKLIST, "split", splits)
    assert set(qs) == {"f01", "f02", "f02.a", "f02.b"} and "criteria" in qs["f02"] and qs["f02.b"] == {"type": "noul", "instructions": "A common error is warned against."}
    p = {"f01": 0.2, "f02": 0.3, "f02.a": 0.1, "f02.b": 0.8}
    assert combine(p, CHECKLIST, "split", splits) == {"f01": 0.2, "f02": 0.8, "f02.joined": 0.3, "f02.a": 0.1, "f02.b": 0.8}
    assert combine(p, CHECKLIST, "statement") == {"f01": 0.2, "f02": 0.3}


def test_split_run(tmp_path):
    splits = {"f02": ["a", "b"]}

    class Split(Engine):
        def many(self, jobs, log, passes, accept, progress):
            r = {"id": jobs[0].id, "usage": {"input_tokens": 1, "output_tokens": 1},
                 "answers": {k: {"type": "noul", "noul": v} for k, v in {"f01": 0.9, "f02": 0.2, "f02.a": 0.75, "f02.b": 0.1}.items()}}
            assert accept(r) and not accept({"id": "x", "answers": {"f01": {"type": "noul", "noul": 0.9}, "f02": {"type": "noul", "noul": 0.2}}})
            return {jobs[0].id: r}

    result = run_jev({"q1": Q["q1"]}, ["q1"], CHECKLIST, "C.", tmp_path, variant="split", engine=Split(), splits=splits, progress=lambda s: None)
    row = result.scores.iloc[0].to_dict()
    assert row["f02"] == 0.75 and row["f02.joined"] == 0.2 and row["f02.a"] == 0.75 and row["f01"] == 0.9


def test_real_splits_cover_known_features():
    import json as _json
    from pathlib import Path
    from qscorer.jev import load_splits, questions_for
    model = _json.loads((Path(__file__).resolve().parents[1] / "models" / "spacing-exp4" / "model.json").read_text())
    splits = load_splits()
    assert splits and all(len(v) >= 2 for v in splits.values())
    qs = questions_for(model["items"], "split")
    assert len(qs) == 45 + sum(len(v) for v in splits.values())


def test_tuned():
    import json as _json
    from pathlib import Path
    from qscorer.jev import calibrated, combine, load_calibration, load_chosen, load_splits, part_ids, questions_for
    model = _json.loads((Path(__file__).resolve().parents[1] / "models" / "spacing-exp4" / "model.json").read_text())
    items, splits, chosen, cal = model["items"], load_splits(), load_chosen(), load_calibration()
    qs = questions_for(items, "tuned")
    assert len(qs) == 45 + sum(len(v) for v in splits.values())
    for f, text in chosen.items():
        assert qs[f] == {"type": "noul", "instructions": text}
    assert "criteria" in qs["f01"] and set(cal) == {it["id"] for it in items}
    p = {k: 0.8 for k in qs}
    out = combine(p, items, "tuned")
    assert all(out[f"{it['id']}.joined"] == 0.8 for it in items) and all(out[f"{f}.split"] == 0.8 for f in splits)
    assert calibrated({"form": "joined", "mode": "threshold", "threshold": 0.85}, 0.8, None) == 0.0
    assert calibrated({"form": "split", "mode": "threshold", "threshold": 0.5}, 0.1, 0.6) == 1.0
    assert abs(calibrated({"form": "joined", "mode": "probability", "a": 1.0, "b": 0.0}, 0.8, None) - 0.8) < 1e-9
