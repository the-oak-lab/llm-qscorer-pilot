"""Stage 0.2 and 0.4 on a hand-made table, and end to end on the synthetic bank."""
import numpy as np
import pandas as pd

from qscorer.config import PairParams, SlopeParams
from qscorer.pairs import classify, cohen_kappa, diagnostics, match, run_pairs, separate
from qscorer.slopes import run_slopes

TABLE = pd.DataFrame({
    "set": ["s1"] * 8, "supported": [True] * 7 + [False],
    "slope": [1.0, 0.9, 0.5, 0.4, 0.1, 0.0, -0.5, 5.0], "slope_sd": [0.2] * 8, "p_positive": [1, 1, .99, .98, .69, .5, .01, 1],
    "first_practice_accuracy": [0.50, 0.90, 0.50, 0.50, 0.52, 0.20, 0.55, 0.50],
}, index=pd.Index(list("abcdefgh"), name="id"))


def test_classify_match_separate():
    pool = classify(TABLE, 0.25)
    assert pool.tolist() == ["high", "high", "", "", "", "low", "low", ""]  # h is unsupported, whatever its slope
    pairs = match(TABLE, pool, caliper=0.10)
    assert pairs.high.tolist() == ["a"] and pairs.low.tolist() == ["g"]       # a-g at 0.05; b is 0.35 from g and 0.70 from f
    assert len(match(TABLE, pool, caliper=None)) == 2                          # without a caliper b-f is formed too
    tested = separate(pairs, TABLE, separation=0.90, min_difference=0.20)
    assert abs(tested.slope_difference[0] - 1.5) < 1e-9 and abs(tested.difference_sd[0] - np.hypot(.2, .2)) < 1e-9
    assert tested.p_separation[0] > 0.999 and bool(tested.selected[0])
    assert not separate(pairs, TABLE, 0.90, min_difference=2.0).selected[0]
    d = diagnostics(TABLE, pool, tested, PairParams())
    assert d["pools"] == {"s1": {"questions": 7, "high": 2, "low": 2}} and d["pairs"]["matched"] == 1 and d["pairs"]["selected"] == 1
    assert d["pairs"]["without_caliper"]["matched"] == 2 and d["alternatives"]["C_sign_probability"] == {"high": 4, "low": 3}
    assert d["alternatives"]["B_meaningful_slope"]["0.2"]["high"] == 3   # a, b, c: P(slope > 0.2) >= 0.9 with sd 0.2
    assert abs(cohen_kappa(pd.Series(list("xyxy")), pd.Series(list("xyxy"))) - 1) < 1e-9


def test_run_pairs_on_the_synthetic_bank(dataset):
    config, _ = dataset
    slope_params = SlopeParams(max_opportunities=5, reliability_splits=0)
    table = run_slopes(config, slope_params).table
    params = PairParams(caliper=1.0, reliability_splits=1)  # q1 and q3 differ by 0.55 in accuracy; one split of the replication check
    result = run_pairs(table, params, config, slope_params)
    q = result.questions.set_index("id")
    assert q.pool.tolist() == ["high", "", "low", "", ""] and q.partner.tolist() == ["q3", "", "q1", "", ""] and q.selected.tolist() == [True, False, True, False, False]
    assert len(result.pairs) == 1 and result.pairs.p_separation[0] > 0.999
    rep = result.diagnostics["replication"]["median"]
    assert rep["questions"] == 3 and rep["pairs_selected_per_half"] >= 0 and np.isfinite(rep["kappa_three_way"])
    try:
        run_pairs(table, params)
    except ValueError as e:
        assert "config" in str(e)
    else:
        raise AssertionError("the replication check without a config must be refused")
