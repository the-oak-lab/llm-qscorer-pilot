"""The power simulation on the shared contrast machinery: the layout indexes the plan, a large effect is found, the null keeps its size."""
import numpy as np

from qscorer.batches import CONDITIONS, BatchParams, plan_batches
from qscorer.contrasts import Layout, differences, sign_flip
from qscorer.power import PowerParams, run_power, simulate
from tests.test_batches import PAIRS


def test_layout_and_simulation():
    plan, _ = plan_batches(PAIRS, BatchParams(rounds=2, seed=0))
    layout = Layout(plan)
    assert layout.slot["mixed", "high"].shape == (2, 15) and (layout.slot["mixed", "high"] >= 0).sum() == 2 * 7 * 2
    assert (layout.mixed_batch >= 0).sum() == 2 * 7 * 2 and len(np.unique(layout.mixed_batch[layout.mixed_batch >= 0])) == 14
    z = simulate(layout, PowerParams(sigma_question=0, sigma_batch=0, sigma_call=0, base_rate=0.5), 0.0, 0.0, np.random.default_rng(0))
    assert set(z) == {"E1", "E3", "E5", "E7", "E9"} and all(v.shape == (len(plan),) for v in z.values())
    conds = list(CONDITIONS)
    z1 = simulate(layout, PowerParams(sigma_question=0, sigma_batch=0, sigma_call=0, base_rate=0.01), 8.0, 0.0, np.random.default_rng(0))
    Zq = np.full((1, len(conds), len(plan)), np.nan)
    for c, v in z1.items():
        Zq[0, conds.index(c)] = v
    Zb = np.full((1, len(conds), layout.n_batches), np.nan)
    d = differences(layout, Zb, Zq, "good")
    assert d["quality"][0].shape == (15, 1) and np.nanmean(d["quality"][0]) > 0.9 and np.nanmean(d["quality_mixed"][0]) > 0.8
    assert d["inversion_question"][0].shape == (30, 1) and (d["inversion_question"][1] >= 0).sum() == 28
    est, p, stat, p_stat = sign_flip(np.ones((20, 1)), np.arange(20), 199, np.random.default_rng(0))
    assert est[0] == 1 and p[0] < 0.02 and stat == 1 and p_stat < 0.02
    assert sign_flip(np.array([1, -1.0] * 10)[:, None], np.arange(20), 199, np.random.default_rng(0))[1][0] > 0.5
    assert sign_flip(np.ones((20, 1)), np.zeros(20), 199, np.random.default_rng(0))[1][0] > 0.4   # one unit: the mean only ever changes sign


def test_run_power_null_and_alternative():
    plan, _ = plan_batches(PAIRS, BatchParams(rounds=3, seed=0))
    params = PowerParams(deltas=[0.0, 4.0], kappas=[4.0], sims=60, flips=199, features=10, enriched=3, seed=1)
    t = run_power(plan, params, progress=lambda s: None)
    null, big, label = t.iloc[0], t.iloc[1], t.iloc[2]
    assert null["quality_0.05"] < 0.2 and null["quality_mixed_0.05"] < 0.15 and null["inversion_0.05"] < 0.15 and null["family_false_discoveries"] < 1.0
    assert big["quality_0.05"] > 0.8 and big["quality_mixed_0.05"] > 0.8 and big["family_true_discoveries"] > 2.0
    assert label["inversion_question_0.05"] > 0.8 and label["quality_0.05"] < 0.2   # a label effect is not a quality effect
    assert label["inversion_0.05"] < 0.2     # the batch-level test cannot see a pure attribution effect: the feature is named either way
