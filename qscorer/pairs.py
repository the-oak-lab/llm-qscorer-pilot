"""Stage 0.2 (`pairs`): high- and low-learning questions and matched pairs; Stage 0.4: the
selection diagnostics.

From the Stage 0.1 table (`slopes.csv`): within each item set, supported questions whose
posterior mean slope is in the top quarter are high-learning candidates and those in the bottom
quarter low-learning candidates. High candidates are matched to low candidates of the same set
on observed first-practice accuracy, closest pairs first, each question used once, no pair
farther apart than the caliper. A pair is kept when the posterior probability that the high
question's slope exceeds the low question's reaches `separation` and the difference of
posterior means reaches `min_difference`. The two posteriors are taken as independent: the
questions share only the student effects, and on the spacing banks the joint posterior changes
the probability by at most 0.0005.

The diagnostics report pool and pair counts per set, the distributions of accuracy and slope
differences and separation probabilities, how many pairs survive at 0.95 and without a caliper,
the pool sizes under the design's alternative definitions (Stage 0.3), the slope's relation to
difficulty, and, with `reliability_splits`, how the labels and pairs replicate across random
halves of the students (each half refitted and the whole procedure redone).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from qscorer.config import DatasetConfig, PairParams, SlopeParams
from qscorer.data import item_sets, load_questions, practice_steps, read_datashop
from qscorer.slopes import question_table

PAIR_COLUMNS = ["set", "high", "low", "accuracy_high", "accuracy_low", "accuracy_difference"]


@dataclass
class PairResult:
    questions: pd.DataFrame   # one row per question: set, pool, partner, selected
    pairs: pd.DataFrame       # one row per matched pair, with its separation test
    diagnostics: dict


def classify(table: pd.DataFrame, quantile: float) -> pd.Series:
    """`high`, `low` or `` per question (index = id): within-set quantiles of the posterior
    mean slope among supported questions."""
    pool = pd.Series("", index=table.index, dtype=object)
    for _, g in table[table.supported].groupby("set"):
        pool[g.index[g.slope >= g.slope.quantile(1 - quantile)]] = "high"
        pool[g.index[g.slope <= g.slope.quantile(quantile)]] = "low"
    return pool


def match(table: pd.DataFrame, pool: pd.Series, caliper: float | None) -> pd.DataFrame:
    """Greedy closest-first matching of high to low questions within each set on first-practice
    accuracy, each question used once; `caliper = None` matches without a limit."""
    rows = []
    acc = table.first_practice_accuracy
    for s, g in table.groupby("set"):
        high, low = g.index[pool[g.index] == "high"], g.index[pool[g.index] == "low"]
        candidates = sorted((abs(acc[h] - acc[l]), h, l) for h in high for l in low if np.isfinite(acc[h]) and np.isfinite(acc[l]))
        used: set[str] = set()
        for d, h, l in candidates:
            if caliper is not None and d > caliper:
                break
            if h in used or l in used:
                continue
            used.update((h, l))
            rows.append({"set": s, "high": h, "low": l, "accuracy_high": acc[h], "accuracy_low": acc[l], "accuracy_difference": d})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def separate(pairs: pd.DataFrame, table: pd.DataFrame, separation: float, min_difference: float) -> pd.DataFrame:
    """Add to each pair the slope difference (high minus low), its posterior sd under
    independence, P(slope_high > slope_low | y), and whether the pair is selected."""
    diff = table.slope[pairs.high].to_numpy(float) - table.slope[pairs.low].to_numpy(float)
    sd = np.hypot(table.slope_sd[pairs.high].to_numpy(float), table.slope_sd[pairs.low].to_numpy(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = norm.cdf(diff / sd)
    return pairs.assign(slope_difference=diff, difference_sd=sd, p_separation=p,
                        selected=np.isfinite(p) & (p >= separation) & (diff >= min_difference))


def select_pairs(table: pd.DataFrame, params: PairParams) -> tuple[pd.Series, pd.DataFrame]:
    """Stage 0.2 on a Stage 0.1 table indexed by id: the pools and the tested pairs."""
    pool = classify(table, params.quantile)
    pairs = separate(match(table, pool, params.caliper), table, params.separation, params.min_difference)
    return pool, pairs


def quantiles(x: pd.Series, qs: tuple[float, ...]) -> dict[str, float]:
    return {str(q): float(x.quantile(q)) for q in qs} if len(x) else {}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = sum(float((a == v).mean()) * float((b == v).mean()) for v in set(a) | set(b))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def diagnostics(table: pd.DataFrame, pool: pd.Series, pairs: pd.DataFrame, params: PairParams) -> dict:
    """Stage 0.4: what the selection looks like, and the alternatives of Stage 0.3."""
    supported = table[table.supported]
    pools = {s: {"questions": int(len(g)), "high": int((pool[g.index] == "high").sum()), "low": int((pool[g.index] == "low").sum())}
             for s, g in supported.groupby("set")}
    free = separate(match(table, pool, None), table, params.separation, params.min_difference)
    sel = pairs[pairs.selected]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = lambda eps: (supported.slope - eps) / supported.slope_sd
    alternatives = {
        "A_quantiles_only": {"high": int((pool == "high").sum()), "low": int((pool == "low").sum())},
        "B_meaningful_slope": {str(eps): {"high": int((norm.cdf(z(eps)) >= .9).sum()), "low": int((norm.sf(z(eps)) >= .9).sum())} for eps in (0.1, 0.2, 0.3)},
        "C_sign_probability": {"high": int((supported.p_positive >= .95).sum()), "low": int((supported.p_positive <= .70).sum())},
    }
    return {
        "params": params.model_dump(mode="json"),
        "pools": pools,
        "pairs": {
            "matched": int(len(pairs)), "selected": int(len(sel)),
            "per_set": {s: {"matched": int(len(g)), "selected": int(g.selected.sum())} for s, g in pairs.groupby("set")},
            "accuracy_difference": quantiles(pairs.accuracy_difference, (.25, .5, .75, 1.0)),
            "slope_difference": quantiles(pairs.slope_difference, (.1, .25, .5, .75, .9)),
            "p_separation": quantiles(pairs.p_separation, (.1, .25, .5, .75, .9)),
            "selected_at_separation": {"0.90": int(((pairs.p_separation >= .90) & (pairs.slope_difference >= params.min_difference)).sum()),
                                       "0.95": int(((pairs.p_separation >= .95) & (pairs.slope_difference >= params.min_difference)).sum())},
            "separated_but_below_min_difference": int(((pairs.p_separation >= params.separation) & (pairs.slope_difference < params.min_difference)).sum()),
            "selected_slope_difference_median": float(sel.slope_difference.median()) if len(sel) else float("nan"),
            "without_caliper": {"matched": int(len(free)), "selected": int(free.selected.sum()),
                                "accuracy_difference_median": float(free.accuracy_difference.median()) if len(free) else float("nan")},
        },
        "alternatives": alternatives,
        "slope_vs_first_practice_accuracy": {
            "spearman": float(supported.slope.corr(supported.first_practice_accuracy, method="spearman")),
            "within_sets": {s: float(g.slope.corr(g.first_practice_accuracy, method="spearman")) for s, g in supported.groupby("set")}},
    }


def split_half_replication(config: DatasetConfig, slope_params: SlopeParams, params: PairParams) -> dict:
    """Refit the model on random halves of the students, redo the pools and pairs in each half,
    and report how the three-way label (high / middle / low) agrees between halves and how often
    a pair selected in one half keeps its direction in the other."""
    ids = list(load_questions(config.questions))
    tx = read_datashop(config.transactions)
    steps, _ = practice_steps(tx, config, slope_params, set(ids))
    sets = pd.Series(item_sets(tx, config, ids))
    students = np.array(sorted(steps[config.student_column].unique()))
    rows = []
    for k in range(params.reliability_splits):
        rng = np.random.default_rng(params.seed + k)
        half = set(rng.permutation(students)[: len(students) // 2])
        tables, pools, pairs = [], [], []
        for part in (steps[steps[config.student_column].isin(half)], steps[~steps[config.student_column].isin(half)]):
            t = question_table(part, config, slope_params)[0]
            t.insert(0, "set", sets.reindex(t.index))
            pool, pr = select_pairs(t, params)
            tables.append(t); pools.append(pool); pairs.append(pr[pr.selected])
        both = pd.concat([pools[0].rename("a"), pools[1].rename("b")], axis=1, join="inner").replace("", "middle")
        both = both[tables[0].supported.reindex(both.index).fillna(False) & tables[1].supported.reindex(both.index).fillna(False)]
        extreme = both[(both.a != "middle") & (both.b != "middle")]
        replicated = []
        for src, other in ((pairs[0], tables[1]), (pairs[1], tables[0])):
            ok = src.high.isin(other.index) & src.low.isin(other.index)
            s = src[ok]
            replicated.append(float((other.slope[s.high].to_numpy() > other.slope[s.low].to_numpy()).mean()) if len(s) else float("nan"))
        rows.append({"split": k, "questions": int(len(both)), "kappa_three_way": cohen_kappa(both.a, both.b),
                     "kappa_high_vs_low": cohen_kappa(extreme.a, extreme.b), "extreme_in_both": int(len(extreme)),
                     "high_low_flips": int(((both.a == "high") & (both.b == "low")).sum() + ((both.a == "low") & (both.b == "high")).sum()),
                     "pairs_selected_per_half": float(np.mean([len(p) for p in pairs])),
                     "direction_replicates": float(np.nanmean(replicated))})
    splits = pd.DataFrame(rows)
    return {"splits": rows, "median": {c: float(splits[c].median()) for c in splits.columns if c != "split"}}


def run_pairs(table: pd.DataFrame, params: PairParams, config: DatasetConfig | None = None,
              slope_params: SlopeParams | None = None) -> PairResult:
    """Stage 0.2 and 0.4 on a Stage 0.1 table (`slopes.csv`, one row per question). The
    replication check refits the model and needs the dataset config and the Stage 0.1 params."""
    t = table.set_index("id") if "id" in table.columns else table
    pool, pairs = select_pairs(t, params)
    partner = pd.concat([pd.Series(pairs.low.to_numpy(), index=pairs.high), pd.Series(pairs.high.to_numpy(), index=pairs.low)])
    selected = pd.concat([pd.Series(pairs.selected.to_numpy(), index=pairs.high), pd.Series(pairs.selected.to_numpy(), index=pairs.low)])
    questions = pd.DataFrame({"set": t.set, "supported": t.supported, "slope": t.slope, "slope_sd": t.slope_sd,
                              "first_practice_accuracy": t.first_practice_accuracy, "pool": pool,
                              "partner": partner.reindex(t.index).fillna(""), "selected": selected.reindex(t.index).fillna(False).astype(bool)})
    questions.index.name = "id"
    diag = diagnostics(t, pool, pairs, params)
    if params.reliability_splits:
        if config is None or slope_params is None:
            raise ValueError("the replication check needs the dataset config and the Stage 0.1 params")
        diag["replication"] = split_half_replication(config, slope_params, params)
    diag["run_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return PairResult(questions=questions.reset_index(), pairs=pairs, diagnostics=diag)
