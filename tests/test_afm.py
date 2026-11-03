"""The slope model on the synthetic bank: the design, the EM loop, and recovery of the planted effects."""
import numpy as np
from conftest import BANK, JUMP, TRUE

from qscorer.afm import build_design, fit_slopes, with_tau2
from qscorer.config import SlopeParams
from qscorer.data import SCHEDULE, practice_steps, read_datashop, to_step_data


def planted(config):
    steps, _ = practice_steps(read_datashop(config.transactions), config, SlopeParams(max_opportunities=5), set(BANK))
    return to_step_data(steps)


def planted_with_schedule(config):
    steps, _ = practice_steps(read_datashop(config.transactions), config, SlopeParams(max_opportunities=5), set(BANK))
    return to_step_data(steps), steps[SCHEDULE].tolist()


def test_design_has_a_shared_mean_and_penalised_deviations(dataset):
    config, _ = dataset
    design = build_design(planted(config), tau2=0.25)
    assert [b.name for b in design.blocks] == ["student", "kc_intercept", "kc_slope", "slope_mean"]
    slope, mean = design.get("kc_slope"), design.get("slope_mean")
    assert (slope.l2 == 4.0).all() and (mean.l2 == 0).all()
    assert np.allclose(np.asarray(slope.matrix.sum(axis=1)).ravel(), mean.matrix.toarray().ravel())  # the mean column is the deviations' sum
    assert not any(c.endswith("|baseline") for c in slope.columns)  # baseline components have no slope
    assert "q4" not in slope.columns  # one opportunity only: its slope column is all zeros and dropped
    assert (with_tau2(design, 1.0).get("kc_slope").l2 == 1.0).all()


def test_em_converges_and_recovers_the_planted_effects(dataset):
    config, _ = dataset
    data = planted(config)
    sf = fit_slopes(data, tau2=0.05)
    assert sf.fit.is_optimal and 1 < len(sf.iterations) < 50 and sf.iterations[-1]["tau2_next"] == sf.tau2
    assert 0.3 < np.sqrt(sf.tau2) < 3.0   # planted slopes 1, 0, -1 per attempt
    s = sf.slopes()
    assert list(s.index) == ["q1", "q2", "q3", "q5"] and s.loc["q5", "separated"] and not s.loc["q1", "separated"]
    assert s.loc["q1", "slope"] > 0 > s.loc["q3", "slope"] and abs(s.loc["q2", "slope"]) < 3 * s.loc["q2", "slope_sd"] + 0.1
    j = sf.jumps()
    for q, (a, _) in TRUE.items():
        assert abs(j.loc[q, "jump"] - JUMP[q]) < 3 * j.loc[q, "jump_sd"] + 0.05, q
    kv = sf.kc_values(data).set_index("KC Name")
    assert kv.loc["q1", "Intercept (logit)"] < 0 and abs(kv.loc["q1|baseline", "Intercept (logit)"] - (TRUE["q1"][0] - JUMP["q1"])) < 3 * j.loc["q1", "jump_sd"] + 0.3
    d = sf.difference_sd(["q1"], ["q3"])[0]
    assert abs(d - np.hypot(s.loc["q1", "slope_sd"], s.loc["q3", "slope_sd"])) < 0.2 * d  # the two slopes are nearly independent
    assert np.isfinite(sf.laplace_log_marginal) and abs(sf.student_effects(data).mean()) < 1e-9


def test_schedule_adjusted_fit(dataset):
    config, _ = dataset
    data, schedule = planted_with_schedule(config)
    design = build_design(data, tau2=0.25, schedule=schedule)
    assert [b.name for b in design.blocks] == ["student", "kc_intercept", "kc_slope", "schedule_jump", "schedule_slope"]
    assert len(design.get("schedule_jump").columns) == 1 and sorted(design.get("schedule_slope").columns) == ["massed", "spaced"]
    assert "slope_mean:mu" not in design.columns
    adj, plain = fit_slopes(data, schedule=schedule, tau2=0.05), fit_slopes(data, tau2=0.05)
    assert adj.fit.is_optimal and set(adj.schedule_terms) == {"jump:massed", "slope:massed", "slope:spaced"} or set(adj.schedule_terms) == {"jump:spaced", "slope:massed", "slope:spaced"}
    assert abs(adj.schedule_terms["slope:massed"] - adj.schedule_terms["slope:spaced"]) < 1.0   # no planted schedule effect
    assert abs(sum(adj.mean_columns.values()) - 1.0) < 1e-9 and abs(adj.mu - plain.mu) < 0.3
    a, b = adj.slopes(), plain.slopes()
    assert list(a.index) == list(b.index) and (a.slope - b.slope).abs().max() < 0.3 and (a.slope_sd > 0).all()
    assert abs(adj.difference_sd(["q1"], ["q3"])[0] - plain.difference_sd(["q1"], ["q3"])[0]) < 0.05
    try:
        build_design(data, 0.25, schedule=schedule[:-1])
    except ValueError as e:
        assert "labels" in str(e)
    else:
        raise AssertionError("a schedule of the wrong length must be refused")
