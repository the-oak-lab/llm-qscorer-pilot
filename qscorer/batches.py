"""Stage 2.1 (`batches`): the batch plan for contrastive feature discovery.

From the Stage 0.2 pairs (`pairs.csv`): in each of `rounds` randomised rounds, every item set's
pairs are grouped two at a time into **mixed** batches (two high- and two low-learning
questions), its high members four at a time into **good** batches, and its low members four at
a time into **bad** batches. Batches never cross item sets. Companions change between rounds:
two questions (or two pairs) batched together in one round are not batched together again
when a grouping that avoids it is found. When a set's count does not divide, the questions (or
pairs) left out of a round are those left out least often so far. The order of the four
questions in a batch is random and recorded; every condition that uses the batch sees that
order.

Conditions (`CONDITIONS`): good batches serve E1 (good features) and E2 (bad features), bad
batches E3 and E4, mixed batches E5 to E10 (unlabelled, true labels, inverted labels, each for
good and bad features). The plan holds the batches; Stage 2.2 builds the calls from it.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

KINDS = {"mixed": 2, "good": 4, "bad": 4}   # units per batch: pairs for mixed batches, questions otherwise
CONDITIONS = {                              # condition: batch kind, feature polarity asked for, labels shown
    "E1": ("good", "good", None), "E2": ("good", "bad", None), "E3": ("bad", "good", None), "E4": ("bad", "bad", None),
    "E5": ("mixed", "good", None), "E6": ("mixed", "bad", None), "E7": ("mixed", "good", "true"), "E8": ("mixed", "bad", "true"),
    "E9": ("mixed", "good", "inverted"), "E10": ("mixed", "bad", "inverted"),
}
COLUMNS = ["round", "set", "kind", "batch", "position", "id", "pool", "pair", "accuracy"]


class BatchParams(BaseModel):
    rounds: int = Field(5, ge=1, description="randomised batching rounds")
    tries: int = Field(200, ge=1, description="shuffles tried per grouping to avoid companions from earlier rounds")
    seed: int = 0


def group(units: list, size: int, seen: set, left_out: Counter, rng: np.random.Generator, tries: int) -> tuple[list[list], list, int]:
    """Partition `units` into groups of `size`. If the count does not divide, the units left out
    are those with the fewest past leave-outs (ties at random). Among `tries` shuffles, keep the
    one with the fewest pairs of units already grouped together (`seen`), and record its
    pairs in `seen`. Returns the groups, the units left out and the number of repeated pairs."""
    n_out = len(units) % size
    out = sorted(units, key=lambda u: (left_out[u], rng.random()))[:n_out] if n_out else []
    kept = [u for u in units if u not in out]
    best, best_repeats = [], 0
    for _ in range(tries):
        order = [kept[i] for i in rng.permutation(len(kept))]
        groups = [order[i:i + size] for i in range(0, len(order), size)]
        repeats = sum(frozenset(ab) in seen for g in groups for ab in combinations(g, 2))
        if not best or repeats < best_repeats:
            best, best_repeats = groups, repeats
        if repeats == 0:
            break
    seen.update(frozenset(ab) for g in best for ab in combinations(g, 2))
    left_out.update(out)
    return best, out, best_repeats


def plan_batches(pairs: pd.DataFrame, params: BatchParams) -> tuple[pd.DataFrame, dict]:
    """The plan (one row per question in a batch, `COLUMNS`) and its summary. `pair` is the
    row number of the pair in `pairs`."""
    pairs = pairs.reset_index(drop=True)
    accuracy = pd.concat([pd.Series(pairs.accuracy_high.to_numpy(), index=pairs.high), pd.Series(pairs.accuracy_low.to_numpy(), index=pairs.low)])
    pair_of = pd.concat([pd.Series(pairs.index, index=pairs.high), pd.Series(pairs.index, index=pairs.low)])
    seen = {k: set() for k in KINDS}
    left_out = {k: Counter() for k in KINDS}
    rows, rounds = [], []
    for r in range(1, params.rounds + 1):
        rng = np.random.default_rng([params.seed, r])
        info = {"round": r, "batches": {k: 0 for k in KINDS}, "left_out": {k: [] for k in KINDS}, "repeated_companions": {k: 0 for k in KINDS}}
        for s, ps in pairs.groupby("set", sort=True):
            units = {"mixed": list(ps.index), "good": list(ps.high), "bad": list(ps.low)}
            for kind, size in KINDS.items():
                groups, out, repeats = group(units[kind], size, seen[kind], left_out[kind], rng, params.tries)
                info["left_out"][kind] += [str(u) for u in out]
                info["repeated_companions"][kind] += repeats
                for g in groups:
                    info["batches"][kind] += 1
                    if kind == "mixed":
                        members = [(pairs.high[p], "high", p) for p in g] + [(pairs.low[p], "low", p) for p in g]
                    else:
                        members = [(q, "high" if kind == "good" else "low", int(pair_of[q])) for q in g]
                    batch = f"r{r}-{s}-{kind}-{info['batches'][kind]:02d}"
                    for position, i in enumerate(rng.permutation(len(members)), 1):
                        q, pool, p = members[i]
                        rows.append({"round": r, "set": s, "kind": kind, "batch": batch, "position": position, "id": q, "pool": pool,
                                     "pair": int(p), "accuracy": float(accuracy[q])})
        rounds.append(info)
    plan = pd.DataFrame(rows, columns=COLUMNS)
    return plan, summarise(plan, rounds, params)


def summarise(plan: pd.DataFrame, rounds: list[dict], params: BatchParams) -> dict:
    per_kind = {k: int((plan[plan.kind == k].groupby("batch").ngroups)) for k in KINDS}
    per_round = {k: per_kind[k] / params.rounds for k in KINDS}
    calls_per_round = sum(per_round[kind] for kind, _, _ in CONDITIONS.values())
    spread = plan.groupby("batch").accuracy.agg(lambda a: a.max() - a.min())
    mixed = plan[plan.kind == "mixed"]
    return {
        "params": params.model_dump(mode="json"),
        "pairs": int(plan.pair.nunique()), "questions": int(plan.id.nunique()),
        "batches": per_kind, "batches_per_round": per_round,
        "per_set_per_round": {s: {k: int(g[g.kind == k].groupby("batch").ngroups / params.rounds) for k in KINDS} for s, g in plan.groupby("set")},
        "calls_per_round_one_per_condition": int(calls_per_round), "calls_total_one_per_condition": int(calls_per_round * params.rounds),
        "rounds": rounds,
        "left_out_per_question_max": {k: int(max(Counter(sum((r["left_out"][k] for r in rounds), [])).values(), default=0)) for k in KINDS},
        "repeated_companions_total": {k: int(sum(r["repeated_companions"][k] for r in rounds)) for k in KINDS},
        "accuracy_range_within_batch": {k: {str(q): float(spread[plan.groupby("batch").kind.first() == k].quantile(q)) for q in (.5, .9, 1.0)} for k in KINDS},
        "high_members_by_position_in_mixed_batches": mixed[mixed.pool == "high"].position.value_counts().sort_index().to_dict(),
    }
