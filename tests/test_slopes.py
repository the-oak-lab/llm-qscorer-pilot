"""Stage 0.1 end to end on the synthetic bank."""
import numpy as np
from conftest import BANK, N_STUDENTS

from qscorer.config import PhaseFilter, SlopeParams
from qscorer.slopes import run_slopes


def test_run_slopes(dataset):
    config, _ = dataset
    result = run_slopes(config, SlopeParams(max_opportunities=5, reliability_splits=2))
    t = result.table.set_index("id")
    assert list(t.index) == BANK and (t.set == "set-1").all()
    assert t.supported.tolist() == [True, True, True, False, False]  # q4 has no second opportunity; q5 is separated
    assert t.loc["q4", "n_second_opportunity"] == 0 and np.isnan(t.loc["q4", "slope"]) and np.isnan(t.loc["q5", "slope"])
    assert t.loc["q1", "p_positive"] > 0.99 and t.loc["q3", "p_positive"] < 0.01 and 0.05 < t.loc["q2", "p_positive"] < 0.95
    assert t.loc["q1", "p_change_over_window"] > 0.3 and t.loc["q3", "p_change_over_window"] < -0.3
    assert t.loc["q1", "p_jump_positive"] > 0.95 and (t.n_baseline == N_STUDENTS).all() and np.isnan(t.loc["q5", "jump"])
    assert abs(t.loc["q1", "p_baseline"] - t.loc["q1", "pretest_accuracy"]) < 0.1
    s = result.summary
    assert s["fit"]["optimal"] and s["questions"] == {"in_file": 5, "with_data": 5, "supported": 3, "sets": {"set-1": 5}}
    assert s["prior"]["tau"] > 0.3 and s["jump"]["questions"] == 3 and s["students"]["n"] == N_STUDENTS
    assert s["reliability"]["median"]["questions"] == 3 and np.isfinite(s["reliability"]["median"]["jump_spearman_brown"])
    assert result.log["questions_in_file"] == 5 and len(result.students) == N_STUDENTS

    no_base = run_slopes(config.model_copy(update={"phases": PhaseFilter(practice=r"^learning\d+$")}), SlopeParams(max_opportunities=5, reliability_splits=0)).table.set_index("id")
    assert no_base.jump.isna().all() and no_base.baseline_logit.isna().all() and (no_base.n_baseline == 0).all()
    assert abs(no_base.loc["q1", "slope"] - t.loc["q1", "slope"]) < 0.15  # the slope is estimated from the same practice rows


def test_schedule_adjusted_run(dataset):
    config, _ = dataset
    plain = run_slopes(config, SlopeParams(max_opportunities=5, reliability_splits=0)).table.set_index("id")
    result = run_slopes(config, SlopeParams(max_opportunities=5, reliability_splits=1, adjust_schedule=True))
    t = result.table.set_index("id")
    assert t.supported.tolist() == plain.supported.tolist() and (t.slope - plain.slope).abs().max() < 0.3
    assert t.loc["q1", "p_positive"] > 0.99 and t.loc["q3", "p_positive"] < 0.01
    sched = result.summary["schedule"]
    assert set(sched["rows"]) == {"massed", "spaced"} and len(sched["terms"]) == 3 and result.summary["params"]["adjust_schedule"]
    assert np.isfinite(result.summary["reliability"]["median"]["slope_spearman_brown"])
    try:
        run_slopes(config.model_copy(update={"schedule_column": None}), SlopeParams(max_opportunities=5, reliability_splits=0, adjust_schedule=True))
    except ValueError as e:
        assert "schedule_column" in str(e)
    else:
        raise AssertionError("adjust_schedule without a schedule column must be refused")
