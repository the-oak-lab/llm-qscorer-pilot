"""Stage 0.2 and 0.4 on a hand-made table, and end to end on the synthetic bank."""
import numpy as np
import pandas as pd

from qscorer.config import PairParams, SlopeParams
from qscorer.pairs import classify, cohen_kappa, diagnostics, match, run_pairs, select_pairs
from qscorer.slopes import run_slopes

TABLE = pd.DataFrame({
    "set": ["s1"] * 8, "supported": [True] * 7 + [False],
    "slope": [1.0, 0.9, 0.5, 0.4, 0.1, 0.0, -0.5, 5.0], "slope_sd": [0.2] * 8, "p_positive": [1, 1, .99, .98, .69, .5, .01, 1],
    "first_practice_accuracy": [0.50, 0.90, 0.50, 0.50, 0.52, 0.20, 0.55, 0.50],
}, index=pd.Index(list("abcdefgh"), name="id"))


def test_classify_match_and_diagnostics():
    pool = classify(TABLE, 0.25)
    assert pool.tolist() == ["high", "high", "", "", "", "low", "low", ""]  # h is unsupported, whatever its slope
    pairs = match(TABLE, pool, 0.10, 0.95, 0.20)
    assert pairs.high.tolist() == ["a"] and pairs.low.tolist() == ["g"]       # a-g at 0.05; b is 0.35 from g and 0.70 from f
    assert abs(pairs.slope_difference[0] - 1.5) < 1e-9 and abs(pairs.difference_sd[0] - np.hypot(.2, .2)) < 1e-9 and pairs.p_separation[0] > 0.999
    assert len(match(TABLE, pool, None, 0.95, 0.20)) == 2                     # without a caliper b-f is formed too
    assert len(match(TABLE, pool, 0.10, 0.95, 2.0)) == 0                      # the minimum difference binds
    d = diagnostics(TABLE, pool, pairs, PairParams(quantile=0.25))
    assert d["pools"] == {"s1": {"questions": 7, "high": 2, "low": 2}} and d["pairs"]["selected"] == 1 and d["pairs"]["per_set"] == {"s1": 1}
    assert d["pairs"]["without_caliper"]["selected"] == 2 and d["pairs"]["at_separation"] == {"0.90": 1, "0.95": 1, "0.99": 1}
    assert d["alternatives"]["C_sign_probability"] == {"high": 4, "low": 3}
    assert d["alternatives"]["B_meaningful_slope"]["0.2"]["high"] == 3   # a, b, c: P(slope > 0.2) >= 0.9 with sd 0.2
    assert abs(cohen_kappa(pd.Series(list("xyxy")), pd.Series(list("xyxy"))) - 1) < 1e-9


def test_match_takes_the_most_pairs_then_the_closest():
    # b-f is the closest pair on accuracy but not separable; a closest-first rule would take it and lose both questions
    t = pd.DataFrame({"set": ["s1"] * 4, "supported": [True] * 4, "slope": [1.0, 0.2, 0.1, -0.5], "slope_sd": [0.2] * 4,
                      "p_positive": [1, .8, .7, 0], "first_practice_accuracy": [0.52, 0.50, 0.50, 0.53]}, index=pd.Index(list("abfg"), name="id"))
    pool = pd.Series(["high", "high", "low", "low"], index=t.index)
    pairs = match(t, pool, 0.10, 0.95, 0.20)
    assert sorted(zip(pairs.high, pairs.low)) == [("a", "f"), ("b", "g")] and (pairs.p_separation >= 0.95).all()
    assert len(match(t, pool, 0.01, 0.95, 0.20)) == 0                          # the caliper still binds
    assert match(t, pool, 0.10, 0.95, 0.80).high.tolist() == ["a"]            # only a-f clears a difference of 0.8
    # among matchings of the same size, the closest: a-f (0.02) + b-g (0.03) beats a-g (0.01) + b-f (0.00) only because b-f is inadmissible,
    # so make both admissible and check the closer set wins
    t2 = t.assign(slope=[1.0, 1.0, -0.5, -0.5])
    assert sorted(zip(match(t2, pool, 0.10, 0.95, 0.20).high, match(t2, pool, 0.10, 0.95, 0.20).low)) == [("a", "g"), ("b", "f")]
    _, pairs = select_pairs(t, PairParams(quantile=0.49, caliper=0.10))
    assert len(pairs) == 2


def test_run_pairs_on_the_synthetic_bank(dataset):
    config, _ = dataset
    slope_params = SlopeParams(max_opportunities=5, reliability_splits=0)
    table = run_slopes(config, slope_params).table
    params = PairParams(quantile=0.25, caliper=1.0, reliability_splits=1)  # q1 and q3 differ by 0.55 in accuracy; one split of the replication check
    result = run_pairs(table, params, config, slope_params)
    q = result.questions.set_index("id")
    assert q.pool.tolist() == ["high", "", "low", "", ""] and q.partner.tolist() == ["q3", "", "q1", "", ""]
    assert len(result.pairs) == 1 and result.pairs.p_separation[0] > 0.999
    rep = result.diagnostics["replication"]["median"]
    assert rep["questions"] == 3 and rep["pairs_selected_per_half"] >= 0 and np.isfinite(rep["kappa_three_way"])
    without = run_pairs(table, params, config.model_copy(update={"exclude": ["q1"]}), slope_params.model_copy(update={"reliability_splits": 0}))
    assert "q1" not in set(without.questions.id) and without.pairs.high.tolist() == ["q2"]   # q1 excluded before the pools: q2 is now the high candidate
    try:
        run_pairs(table, params)
    except ValueError as e:
        assert "config" in str(e)
    else:
        raise AssertionError("the replication check without a config must be refused")
