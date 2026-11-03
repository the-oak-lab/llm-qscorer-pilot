"""The batch plan: within-set batches, each question once per round and kind, rotation of the units left out, fixed order."""
import numpy as np
import pandas as pd

from qscorer.batches import CONDITIONS, BatchParams, group, plan_batches

PAIRS = pd.DataFrame({"set": ["s1"] * 9 + ["s2"] * 6, "high": [f"h{i}" for i in range(15)], "low": [f"l{i}" for i in range(15)],
                      "accuracy_high": np.linspace(0.3, 0.8, 15), "accuracy_low": np.linspace(0.3, 0.8, 15) + 0.02})


def test_plan():
    plan, s = plan_batches(PAIRS, BatchParams(rounds=3, seed=1))
    assert set(plan.columns) == {"round", "set", "kind", "batch", "position", "id", "pool", "pair", "accuracy"}
    for (r, kind), g in plan.groupby(["round", "kind"]):
        assert g.groupby("batch").size().eq(4).all() and g.id.is_unique            # four questions per batch, each question once
        assert (g.groupby("batch").set.nunique() == 1).all()                        # never across sets
        assert sorted(g.groupby("batch").position.apply(sorted).iloc[0]) == [1, 2, 3, 4]
    mixed = plan[plan.kind == "mixed"]
    assert (mixed.groupby("batch").pool.value_counts().unstack() == 2).all().all()  # two high, two low
    assert (mixed.groupby("batch").pair.nunique() == 2).all()                       # from two pairs
    assert (plan[plan.kind == "good"].pool == "high").all() and (plan[plan.kind == "bad"].pool == "low").all()
    # s1: 9 pairs -> 4 mixed batches, 1 pair out; 9 members -> 2 batches, 1 out. s2: 6 -> 3 mixed; 6 -> 1 batch, 2 out
    assert s["per_set_per_round"] == {"s1": {"mixed": 4, "good": 2, "bad": 2}, "s2": {"mixed": 3, "good": 1, "bad": 1}}
    assert s["batches_per_round"] == {"mixed": 7, "good": 3, "bad": 3} and s["calls_per_round_one_per_condition"] == 7 * 6 + 3 * 2 + 3 * 2
    assert s["left_out_per_question_max"]["mixed"] == 1 and s["left_out_per_question_max"]["good"] == 1  # sitting out rotates
    assert s["repeated_companions_total"]["mixed"] == 0                                                 # companions change between rounds
    assert sum(s["high_members_by_position_in_mixed_batches"].values()) == 2 * 7 * 3
    assert plan.equals(plan_batches(PAIRS, BatchParams(rounds=3, seed=1))[0])                           # reproducible


def test_group_prefers_new_companions_and_rotates_leave_outs():
    from collections import Counter
    rng = np.random.default_rng(0)
    seen, out = set(), Counter()
    g1, o1, rep1 = group(list("abcde"), 2, seen, out, rng, 50)
    g2, o2, rep2 = group(list("abcde"), 2, seen, out, rng, 50)
    assert len(g1) == 2 and len(o1) == 1 and rep1 == 0 and rep2 == 0 and o1 != o2
    assert CONDITIONS["E9"] == ("mixed", "good", "inverted")
