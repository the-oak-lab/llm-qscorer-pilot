"""The power simulation: the layout indexes the plan, the null keeps its size, a large effect is found."""
import numpy as np

from qscorer.batches import BatchParams, plan_batches
from qscorer.power import Layout, PowerParams, benjamini_hochberg, differences, run_power, sign_flip_p, simulate
from tests.test_batches import PAIRS


def test_layout_and_simulation():
    plan, _ = plan_batches(PAIRS, BatchParams(rounds=2, seed=0))
    layout = Layout(plan)
    assert layout.slot["mixed", "high"].shape == (2, 15) and (layout.slot["mixed", "high"] >= 0).sum() == 2 * 7 * 2
    z = simulate(layout, PowerParams(sigma_question=0, sigma_batch=0, sigma_call=0, base_rate=0.5), 0.0, 0.0, np.random.default_rng(0))
    assert set(z) == {"E1", "E3", "E5", "E7", "E9"} and all(v.shape == (len(plan),) for v in z.values())
    d = differences(layout, z)
    assert d["quality"][0].shape == (15,) and np.isfinite(d["quality"][0]).all() and d["inversion"][0].shape == (2, 15)
    assert (d["inversion"][1] >= 0).sum() == 2 * 7 * 2 and len(np.unique(d["inversion"][1][d["inversion"][1] >= 0])) == 14   # one unit per mixed batch
    z1 = simulate(layout, PowerParams(sigma_question=0, sigma_batch=0, sigma_call=0, base_rate=0.01), 8.0, 0.0, np.random.default_rng(0))
    assert differences(layout, z1)["quality"][0].mean() > 0.9        # a huge quality effect: high members attributed, low members not
    assert np.nanmean(differences(layout, z1)["quality_mixed"][0]) > 0.8
    assert sign_flip_p(np.ones(20), np.arange(20), 199, np.random.default_rng(0)) < 0.02
    assert sign_flip_p(np.array([1, -1.0] * 10), np.arange(20), 199, np.random.default_rng(0)) > 0.5
    assert sign_flip_p(np.ones(20), np.zeros(20), 199, np.random.default_rng(0)) > 0.4   # one unit: the mean only ever changes sign
    assert benjamini_hochberg(np.array([0.001, 0.2, 0.9, 0.004]), 0.05).tolist() == [True, False, False, True]


def test_run_power_null_and_alternative():
    plan, _ = plan_batches(PAIRS, BatchParams(rounds=3, seed=0))
    params = PowerParams(deltas=[0.0, 4.0], kappas=[4.0], sims=60, flips=199, features=10, enriched=3, seed=1)
    t = run_power(plan, params, progress=lambda s: None)
    null, big, label = t.iloc[0], t.iloc[1], t.iloc[2]
    assert null["quality_0.05"] < 0.2 and null["quality_mixed_0.05"] < 0.15 and null["inversion_0.05"] < 0.15 and null["family_false_discoveries"] < 1.0
    assert big["quality_0.05"] > 0.8 and big["quality_mixed_0.05"] > 0.8 and big["family_true_discoveries"] > 2.0
    assert label["inversion_0.05"] > 0.8 and label["labels_0.05"] > 0.8 and label["quality_0.05"] < 0.2  # a label effect is not a quality effect
