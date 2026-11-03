"""The analysis on simulated discovery tables with known effects: a feature enriched in high questions is found by the
quality tests, a null feature is not, labels move the inversion test, and the multiplicity helpers behave."""
import numpy as np
import pandas as pd

from qscorer.batches import CONDITIONS, BatchParams, plan_batches
from qscorer.contrasts import ContrastParams, Layout, analyse, benjamini_hochberg, holm, outcomes, q_values, raw_mapping
from qscorer.power import PowerParams, simulate
from tests.test_batches import PAIRS

PLAN, _ = plan_batches(PAIRS, BatchParams(rounds=3, seed=0))


def discovery_tables(plan, effects: dict[str, tuple[float, float]], samples=2, seed=0):
    """Simulated calls, features and attributions: one raw feature per canonical name per call when attributed anywhere."""
    layout = Layout(plan)
    rng = np.random.default_rng(seed)
    conds = list(CONDITIONS)
    calls, feats, attrs = [], [], []
    for sample in range(samples):
        z = {name: simulate(layout, PowerParams(sigma_question=0.5, sigma_batch=0.3, sigma_call=0.3, base_rate=0.3), d, k, rng) for name, (d, k) in effects.items()}
        for b, g in plan.groupby("batch"):
            kind = g.kind.iloc[0]
            for c in conds:
                if CONDITIONS[c][0] != kind:
                    continue
                zc = c if c in next(iter(z.values())) else {"E2": "E1", "E4": "E3", "E6": "E5", "E8": "E7", "E10": "E9"}[c]   # bad conditions mirror the good ones
                n = 0
                for name in effects:
                    rows = [(int(r.position), r.id) for r in g.itertuples() if z[name][zc][r.Index] > 0.5]
                    if not rows:
                        continue
                    n += 1
                    feats.append({"batch": b, "condition": c, "sample": sample, "feature": n, "name": name})
                    for pos, qid in rows:
                        attrs.append({"batch": b, "condition": c, "sample": sample, "feature": n, "name": name, "position": pos, "id": qid, "exhibits": True})
                calls.append({"batch": b, "condition": c, "sample": sample, "n_features": n})
    return pd.DataFrame(calls), pd.DataFrame(feats), pd.DataFrame(attrs)


def test_outcomes_and_analysis():
    calls, features, attributions = discovery_tables(PLAN, {"enriched": (3.0, 0.0), "null": (0.0, 0.0), "label": (0.0, 3.0)})
    mapping = raw_mapping(features)
    layout = Layout(PLAN)
    ids, Zb, Zq = outcomes(layout, calls, features, attributions, mapping, min_calls=1)
    assert ids == ["enriched", "label", "null"] and Zb.shape == (3, 10, layout.n_batches) and Zq.shape == (3, 10, len(PLAN))
    e5 = list(CONDITIONS).index("E5")
    assert np.nanmax(Zb[:, e5]) <= 1 and np.isnan(Zq[0, list(CONDITIONS).index("E1"), layout.mixed]).all()   # E1 never asked on mixed batches
    table, omnibus = analyse(PLAN, calls, features, attributions, mapping, ContrastParams(flips=499, boot=200, min_calls=1))
    t = table.set_index(["side", "test", "feature"])
    assert t.loc[("good", "quality_mixed", "enriched"), "p"] < 0.01 and t.loc[("good", "quality_mixed", "enriched"), "estimate"] > 0.3
    assert t.loc[("good", "quality", "enriched"), "p"] < 0.01 and t.loc[("good", "quality_mixed", "null"), "p"] > 0.05
    assert t.loc[("good", "inversion_question", "label"), "p"] < 0.01 and t.loc[("good", "labels", "label"), "p"] < 0.01
    assert t.loc[("good", "inversion", "label"), "p"] == 1.0      # a pure attribution effect never changes whether the feature is named for the batch
    assert t.loc[("good", "inversion", "null"), "p"] > 0.05
    assert t.loc[("good", "quality_mixed", "enriched"), "ci_low"] > 0 and t.loc[("good", "quality_mixed", "null"), "ci_low"] < 0 < t.loc[("good", "quality_mixed", "null"), "ci_high"]
    o = omnibus.set_index(["side", "test"])
    assert o.loc[("good", "quality_mixed"), "p"] < 0.01 and o.loc[("good", "inversion_question"), "p"] < 0.01 and bool(o.loc[("good", "quality_mixed"), "holm_rejected"])
    assert o.primary.sum() == 4 and o.holm_rejected.notna().sum() == 4
    assert set(table.test) == {"quality", "quality_batch", "quality_mixed", "inversion", "inversion_question", "labels", "composition"} and set(table.side) == {"good", "bad"}
    assert t.loc[("bad", "quality_mixed", "enriched"), "estimate"] < 0   # the bad side favours the low member, so an enriched-in-high feature goes negative


def test_multiplicity_helpers():
    assert benjamini_hochberg(np.array([0.001, 0.2, np.nan, 0.004])).tolist() == [True, False, False, True]
    q = q_values(np.array([0.01, 0.04, 0.03, np.nan]))
    assert abs(q[0] - 0.03) < 1e-9 and abs(q[1] - 0.04) < 1e-9 and abs(q[2] - 0.04) < 1e-9 and np.isnan(q[3])
    assert holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}) == {"a": True, "b": False, "c": False, "d": False}   # 0.02 > 0.05 / 3 stops the step-down
    assert holm({"a": 0.001, "b": 0.012, "c": 0.02, "d": 0.5}) == {"a": True, "b": True, "c": True, "d": False}
    assert holm({"a": 0.02, "b": 0.02}) == {"a": True, "b": True} and holm({"a": 0.03, "b": 0.03}) == {"a": False, "b": False}
