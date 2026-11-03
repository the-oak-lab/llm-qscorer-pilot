"""Shared fixture: a small synthetic DataShop transaction file with known learning rates."""
import json

import numpy as np
import pandas as pd
import pytest

from qscorer.config import DatasetConfig, PhaseFilter, TimedOutMarker

N_STUDENTS, N_SESSIONS = 80, 5
TRUE = {"q1": (-1.5, 1.0), "q2": (0.0, 0.0), "q3": (1.0, -1.0)}  # first-practice intercept, slope in logits per attempt
JUMP = {"q1": 1.0, "q2": 0.0, "q3": -0.5}  # pretest logit = intercept - jump
BANK = ["q1", "q2", "q3", "q4", "q5"]


def expit(z):
    return 1 / (1 + np.exp(-z))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Students answer q1..q5 at a pretest and in five learning sessions; q9 is not a question
    in the bank; q4 is shown once only; q5 is always correct. In session 1 every student sees
    q1 twice (two presentations), and one student re-submits q2 within one presentation."""
    rng = np.random.default_rng(1)
    rows = []
    def add(student, phase, q, start, t, outcome, inp="x", timed_out="FALSE"):
        rows.append({"Anon Student Id": student, "Session Id": f"{student}-{phase}", "Time": t, "Level (Phase)": phase,
                     "Problem Name": f"stem {q}", "Problem Start Time": start, "Step Name": q,
                     "Outcome": outcome, "Input": inp, "CF (Timed Out)": timed_out})
    for s in range(N_STUDENTS):
        sid = f"s{s:03d}"
        for q in BANK + ["q9"]:
            p_pre = {"q4": .5, "q5": 1.0, "q9": 0.0}.get(q, expit(TRUE.get(q, (0, 0))[0] - JUMP.get(q, 0)))
            add(sid, "pretest1", q, "2026-01-01 10:00:00", "2026-01-01 10:00:05", "CORRECT" if rng.random() < p_pre else "INCORRECT")
        if s == 0:  # a repeated pretest (a software error in exp3): only the first counts as the baseline
            add(sid, "pretest2", "q2", "2026-01-01 11:00:00", "2026-01-01 11:00:05", "CORRECT")
        opp = {q: 0 for q in ["q1", "q2", "q3"]}
        for k in range(1, N_SESSIONS + 1):
            day = f"2026-01-{k + 1:02d}"
            for q in ["q1", "q2", "q3"]:
                a, b = TRUE[q]
                y = rng.random() < expit(a + b * opp[q]); opp[q] += 1
                add(sid, f"learning{k}", q, f"{day} 10:0{k}:00", f"{day} 10:0{k}:05", "CORRECT" if y else "INCORRECT")
            if k == 1:  # a second presentation of q1 in the same session, a minute later
                y = rng.random() < expit(TRUE["q1"][0] + TRUE["q1"][1] * opp["q1"]); opp["q1"] += 1
                add(sid, "learning1", "q1", f"{day} 10:02:00", f"{day} 10:02:05", "CORRECT" if y else "INCORRECT")
                add(sid, "learning1", "q4", f"{day} 10:03:00", f"{day} 10:03:05", "CORRECT" if rng.random() < .5 else "INCORRECT")
            add(sid, f"learning{k}", "q5", f"{day} 10:09:00", f"{day} 10:09:05", "CORRECT")
            add(sid, f"learning{k}", "q9", f"{day} 10:08:00", f"{day} 10:08:05", "INCORRECT")
        if s == 0:  # a re-submission within one presentation, and a no-answer timeout
            add(sid, "learning2", "q2", "2026-01-03 10:02:00", "2026-01-03 10:02:09", "CORRECT")
            add(sid, "learning3", "q3", "2026-01-04 10:07:00", "2026-01-04 10:07:45", "INCORRECT", inp="", timed_out="TRUE")
    tx = pd.DataFrame(rows)
    tx["Condition Name"] = np.where(tx["Anon Student Id"].str[-1].astype(int) % 2 == 0, "spaced", "massed")  # a schedule per student, no planted effect
    start = pd.to_datetime(tx["Problem Start Time"])
    tx["Problem View"] = start.groupby([tx["Anon Student Id"], tx["Problem Name"]]).rank(method="dense").astype(int).astype(str)
    root = tmp_path_factory.mktemp("data")
    tx.to_csv(root / "tx.txt", sep="\t", index=False)
    with open(root / "q.jsonl", "w") as f:
        for q in BANK:
            f.write(json.dumps({"id": q, "question": {"stem": f"stem {q}", "choices": []}, "answerKey": "a", "feedback-text": ""}) + "\n")
    config = DatasetConfig(name="synthetic", transactions=root / "tx.txt", questions=root / "q.jsonl",
                           phases=PhaseFilter(practice=r"^learning\d+$", baseline=r"^pretest\d*$"), timed_out=TimedOutMarker(),
                           schedule_column="Condition Name")
    return config, tx
