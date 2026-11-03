"""DataShop files to opportunities: the rollup rule, the cap, timeouts, item sets, observed accuracy."""
import numpy as np
from conftest import BANK, N_STUDENTS

from qscorer.config import SlopeParams
from qscorer.data import item_sets, observed_accuracy, practice_steps, read_datashop


def test_practice_steps_follow_datashop_rule(dataset):
    config, tx = dataset
    steps, log = practice_steps(read_datashop(config.transactions), config, SlopeParams(max_opportunities=None), set(BANK))
    assert log["rows_about_questions"] == (tx["Step Name"] != "q9").sum()
    assert log["rows_in_practice"] == ((tx["Step Name"] != "q9") & tx["Level (Phase)"].str.startswith("learning")).sum()
    assert log["rows_in_baseline"] == ((tx["Step Name"] != "q9") & tx["Level (Phase)"].str.startswith("pretest")).sum()
    # the re-submission is a second attempt of one presentation, not an opportunity
    assert log["opportunities"] == log["rows_in_practice"] + log["rows_in_baseline"] - 1
    assert log["baseline_repeats_dropped"] == 1 and (steps["Role"].eq("baseline") & steps["Step Name"].eq("q2")).sum() == N_STUDENTS
    base = steps["Role"].eq("baseline")
    assert (steps.loc[base, "KC (Question)"] == steps.loc[base, "Step Name"] + "|baseline").all()
    assert steps.loc[base, "Opportunity (Question)"].eq("1").all()
    s0 = steps[(steps["Anon Student Id"] == "s000") & steps["Role"].eq("practice")]
    opp = s0.set_index(["Step Name", "Level (Phase)"])["Opportunity (Question)"].astype(int)
    assert np.atleast_1d(opp[("q1", "learning1")]).tolist() == [1, 2]       # two presentations in one session are two opportunities
    assert np.atleast_1d(opp[("q1", "learning2")]).tolist() == [3] and np.atleast_1d(opp[("q1", "learning5")]).tolist() == [6]
    assert np.atleast_1d(opp[("q2", "learning1")]).tolist() == [1]          # counting starts in the practice phase, not at the pretest
    assert s0[s0["Step Name"] == "q4"]["Opportunity (Question)"].tolist() == ["1"]


def test_cap_and_timeouts(dataset):
    config, _ = dataset
    tx = read_datashop(config.transactions)
    capped, log = practice_steps(tx, config, SlopeParams(max_opportunities=2), set(BANK))
    assert capped["Opportunity (Question)"].astype(int).max() == 2
    assert log["opportunities_within_cap"] == len(capped) and capped["Role"].eq("baseline").sum() == 5 * N_STUDENTS  # baseline rows are never capped
    kept, _ = practice_steps(tx, config, SlopeParams(timeouts="keep", max_opportunities=None), set(BANK))
    dropped, log = practice_steps(tx, config, SlopeParams(timeouts="drop", max_opportunities=None), set(BANK))
    assert log["no_answer_timeouts_dropped"] == 1 and len(dropped) == len(kept) - 1
    timeout = kept[kept["First Transaction Time"] == "2026-01-04 10:07:45"]
    assert len(timeout) == 1 and timeout["First Attempt"].iloc[0] == "incorrect"  # kept as an incorrect first attempt


def test_item_sets(dataset):
    config, _ = dataset
    tx = read_datashop(config.transactions)
    tx["Class"] = np.where(tx["Anon Student Id"].str[-1].astype(int) % 2 == 0, "even", "odd")  # two classes, every question in both
    sets = item_sets(tx, config.model_copy(update={"set_column": "Class"}), ["q1", "q2", "q9", "q7"])
    assert sets["q1"] == sets["q2"] == "set-1" and sets["q9"] == "set-1"  # q9 is in the log, just not in the bank
    assert sets["q7"] == "set-2"  # never in the log: its own (empty) set
    assert set(item_sets(tx, config, BANK).values()) == {"set-1"}  # no set column: one set


def test_observed_accuracy(dataset):
    config, _ = dataset
    steps, _ = practice_steps(read_datashop(config.transactions), config, SlopeParams(), set(BANK))
    acc = observed_accuracy(steps, config, BANK)
    assert (acc.n_pretest == N_STUDENTS).all() and acc.loc["q5", "first_practice_accuracy"] == 1.0
    assert acc.loc["q4", "n_first_practice"] == N_STUDENTS and 0.3 < acc.loc["q4", "first_practice_accuracy"] < 0.7
    assert abs(acc.loc["q1", "pretest_accuracy"] - 1 / (1 + np.exp(2.5))) < 0.1
