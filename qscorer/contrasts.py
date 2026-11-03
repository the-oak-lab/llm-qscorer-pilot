"""Stage 2.3 scoring and 2.4 inference (`contrasts`): from canonical features to tested contrasts.

Input: the batch plan, the discovery tables (`calls.csv`, `features.csv`, `attributions.csv`)
and a **mapping** from each raw feature of each call (batch, condition, sample, feature) to a
canonical feature id, or to nothing. The mapping is the output of the normalisation, whatever
produced it: a cut of a clustering tree, a Gemini assignment, or, for a smoke test, the raw
name itself. The module never asks where it came from, so the analysis can be rerun at any
granularity.

Outcomes, per canonical feature f:

- batch level, Z_fbc: the share of the calls on batch b under condition c that name f;
- question level, Z_fjbc: the share of those calls that attribute f to question j (`exhibits`).

Contrasts, each a mean of within-unit differences with a sign-flip permutation p-value
(two-sided), flipped by the unit that is exchangeable under the null; `TESTS` lists them. The
good-feature tests use the good-feature conditions and the high members as the favoured
side; the bad-feature tests mirror them (bad-feature conditions, low members favoured).

| test | compares | level | unit | exact |
|---|---|---|---|---|
| quality | E1 vs E3: the high member in its good batches against the low member in its bad batches | question | pair | no (pairs share batches) |
| quality_batch | E1 vs E3: batches naming f | batch | batch | no (two-sample) |
| quality_mixed | within E5: high against low member of the same batch and call | question | pair | yes |
| inversion | E7 vs E9: f named on the same batch | batch | batch | yes |
| inversion_question | E7 vs E9: within-pair difference under true minus inverted labels | question | batch | yes |
| labels | E7 vs E5: f named on the same batch | batch | batch | yes |
| composition | E5 vs E1/E3: the pair's members in their mixed batch against their homogeneous batches | question | pair | no |

Per test, every feature gets an estimate, a bootstrap interval over units, a p-value and a
Benjamini-Hochberg q-value across features; an **omnibus** row tests the whole feature profile
with the L1 statistic sum_f |mean_u D_uf| under the same flips. The primary family is the
omnibus quality_mixed and inversion tests for good and bad features, Holm-adjusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from qscorer.batches import CONDITIONS

GOOD = {"homogeneous_favoured": "E1", "homogeneous_other": "E3", "mixed": "E5", "true": "E7", "inverted": "E9", "favoured": "high", "other": "low"}
BAD = {"homogeneous_favoured": "E4", "homogeneous_other": "E2", "mixed": "E6", "true": "E8", "inverted": "E10", "favoured": "low", "other": "high"}
SIDES = {"good": GOOD, "bad": BAD}
TESTS = ("quality", "quality_batch", "quality_mixed", "inversion", "inversion_question", "labels", "composition")
PRIMARY = ("quality_mixed", "inversion")
KEY = ["batch", "condition", "sample", "feature"]


class ContrastParams(BaseModel):
    flips: int = Field(9999, ge=1, description="sign flips per permutation test")
    boot: int = Field(2000, ge=0, description="bootstrap resamples of the units for the intervals; 0 skips")
    min_calls: int = Field(10, ge=1, description="a canonical feature named in fewer calls is left out")
    seed: int = 0


class Layout:
    """The plan as index arrays: for each kind and pool, the plan row of each pair's member in
    each round (-1 when the pair sits out); each row's question, batch and pair index; the
    mixed batch of each pair in each round."""

    def __init__(self, plan: pd.DataFrame):
        self.plan = plan.reset_index(drop=True)
        self.pairs = np.sort(self.plan.pair.unique())
        self.rounds = np.sort(self.plan["round"].unique())
        self.question, self.questions = pd.factorize(self.plan.id)
        self.batch, self.batches = pd.factorize(self.plan.batch)
        self.n_questions, self.n_batches = len(self.questions), len(self.batches)
        self.high = (self.plan.pool == "high").to_numpy()
        self.mixed = (self.plan.kind == "mixed").to_numpy()
        self.slot = {}
        pair_pos = pd.Series(np.arange(len(self.pairs)), index=self.pairs)
        round_pos = pd.Series(np.arange(len(self.rounds)), index=self.rounds)
        for kind in ("mixed", "good", "bad"):
            for pool in ("high", "low"):
                rows = self.plan[(self.plan.kind == kind) & (self.plan.pool == pool)]
                slot = np.full((len(self.rounds), len(self.pairs)), -1)
                slot[round_pos[rows["round"]].to_numpy(), pair_pos[rows.pair].to_numpy()] = rows.index.to_numpy()
                self.slot[kind, pool] = slot
        s = self.slot["mixed", "high"]
        self.mixed_batch = np.where(s >= 0, self.batch[np.maximum(s, 0)], -1)   # (rounds, pairs)
        self.batch_kind = self.plan.groupby("batch", sort=False).kind.first().reindex(self.batches).to_numpy()

    def pick(self, z: np.ndarray, kind: str, pool: str) -> np.ndarray:
        """z[..., row] at each pair's member in each round, NaN where the pair sits out: shape (..., rounds, pairs)."""
        slot = self.slot[kind, pool]
        out = z[..., np.maximum(slot, 0)]
        return np.where(slot >= 0, out, np.nan)


def nanmean0(v: np.ndarray, axis: int) -> np.ndarray:
    n = np.isfinite(v).sum(axis)
    return np.where(n > 0, np.nansum(v, axis) / np.maximum(n, 1), np.nan)


def outcomes(layout: Layout, calls: pd.DataFrame, features: pd.DataFrame, attributions: pd.DataFrame, mapping: pd.DataFrame,
             min_calls: int) -> tuple[list, np.ndarray, np.ndarray]:
    """The canonical feature ids kept, Zb (features x conditions x batches) and Zq (features x
    conditions x plan rows), each the share of answered calls on the batch and condition that
    name, or attribute, the feature; NaN where no call was answered."""
    m = mapping.dropna(subset=["canonical"])
    m = m[m.canonical.astype(str).str.len() > 0]
    named = features.merge(m[KEY + ["canonical"]], on=KEY, how="inner")
    counts = named.groupby("canonical").size()
    ids = sorted(counts[counts >= min_calls].index)
    conds = list(CONDITIONS)
    cond_pos = {c: i for i, c in enumerate(conds)}
    id_pos = {f: i for i, f in enumerate(ids)}
    batch_pos = pd.Series(np.arange(layout.n_batches), index=layout.batches)
    answered = calls[calls.n_features.notna()]
    denom_b = np.zeros((len(conds), layout.n_batches))
    np.add.at(denom_b, (answered.condition.map(cond_pos).to_numpy(), batch_pos[answered.batch].to_numpy()), 1)
    Zb = np.zeros((len(ids), len(conds), layout.n_batches))
    named = named[named.canonical.isin(id_pos)].drop_duplicates(KEY + ["canonical"])   # a call naming f twice counts once
    np.add.at(Zb, (named.canonical.map(id_pos).to_numpy(), named.condition.map(cond_pos).to_numpy(), batch_pos[named.batch].to_numpy()), 1)
    row_pos = pd.Series(np.arange(len(layout.plan)), index=pd.MultiIndex.from_frame(layout.plan[["batch", "id"]]))
    attributed = attributions[attributions.exhibits].merge(m[KEY + ["canonical"]], on=KEY, how="inner")
    attributed = attributed[attributed.canonical.isin(id_pos)].drop_duplicates(["batch", "condition", "sample", "canonical", "id"])
    Zq = np.zeros((len(ids), len(conds), len(layout.plan)))
    np.add.at(Zq, (attributed.canonical.map(id_pos).to_numpy(), attributed.condition.map(cond_pos).to_numpy(),
                   row_pos[list(zip(attributed.batch, attributed.id))].to_numpy()), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Zb = np.where(denom_b > 0, Zb / np.maximum(denom_b, 1), np.nan)
        denom_q = denom_b[:, layout.batch]
        Zq = np.where(denom_q > 0, Zq / np.maximum(denom_q, 1), np.nan)
    return ids, Zb, Zq


def differences(layout: Layout, Zb: np.ndarray, Zq: np.ndarray, side: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per test: the within-unit differences, shape (observations, features), and the flipping
    unit of each observation. `Zb` and `Zq` are indexed (feature, condition, batch or row)."""
    s = SIDES[side]
    conds = list(CONDITIONS)
    c = lambda name: conds.index(s[name])
    F = Zq.shape[0]
    pair_unit = np.arange(len(layout.pairs))
    q = lambda cond, kind, pool: layout.pick(Zq[:, c(cond)], kind, pool)          # (F, rounds, pairs)
    avg = lambda v: nanmean0(v, axis=1)                                           # over rounds: (F, pairs)
    m = lambda cond, pool: q(cond, "mixed", pool)
    within = lambda cond: m(cond, s["favoured"]) - m(cond, s["other"])
    mixed_batches = np.flatnonzero(layout.batch_kind == "mixed")
    homogeneous = lambda name: np.flatnonzero(layout.batch_kind == CONDITIONS[s[name]][0])
    b = lambda name, which: Zb[:, c(name)][:, which]                             # (F, batches)
    out = {
        "quality": ((avg(q("homogeneous_favoured", CONDITIONS[s["homogeneous_favoured"]][0], s["favoured"]))
                     - avg(q("homogeneous_other", CONDITIONS[s["homogeneous_other"]][0], s["other"]))).T, pair_unit),
        "quality_mixed": (avg(within("mixed")).T, pair_unit),
        "inversion": ((b("true", mixed_batches) - b("inverted", mixed_batches)).T, mixed_batches),
        "labels": ((b("true", mixed_batches) - b("mixed", mixed_batches)).T, mixed_batches),
    }
    # question-level inversion and composition keep the round: one observation per pair and round
    inv = (within("true") - within("inverted")).reshape(F, -1).T
    out["inversion_question"] = (inv, layout.mixed_batch.reshape(-1))
    comp = ((m("mixed", "high") + m("mixed", "low")) / 2
            - (q("homogeneous_favoured", CONDITIONS[s["homogeneous_favoured"]][0], s["favoured"])
               + q("homogeneous_other", CONDITIONS[s["homogeneous_other"]][0], s["other"])) / 2).reshape(F, -1).T
    out["composition"] = (comp, np.tile(pair_unit, len(layout.rounds)))
    # batch-level quality is a two-sample comparison: favoured homogeneous batches against the other kind
    fav, oth = homogeneous("homogeneous_favoured"), homogeneous("homogeneous_other")
    two = np.concatenate([b("homogeneous_favoured", fav), -b("homogeneous_other", oth)], axis=1).T   # signs make a sum into a difference
    out["quality_batch"] = (two, np.concatenate([fav, oth]))
    return out


def sign_flip(D: np.ndarray, unit: np.ndarray, flips: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per feature (columns of D): the estimate (mean over observations, NaN left out) and the
    two-sided sign-flip p-value; and the omnibus L1 statistic over features with its p-value.
    Observations with the same unit flip together; a unit of -1 is left out."""
    keep = unit >= 0
    D, unit = D[keep], unit[keep]
    codes, units = pd.factorize(unit)
    valid = np.isfinite(D)
    n = valid.sum(0)
    D0 = np.where(valid, D, 0.0)
    est = np.where(n > 0, D0.sum(0) / np.maximum(n, 1), np.nan)
    signs = rng.choice([-1.0, 1.0], size=(flips, len(units)))
    flipped = (signs[:, codes] @ D0) / np.maximum(n, 1)                          # (flips, features)
    p = (1 + (np.abs(flipped) >= np.abs(est) - 1e-12).sum(0)) / (flips + 1)
    p = np.where(n > 0, p, np.nan)
    ok = np.isfinite(est)
    omnibus = float(np.abs(est[ok]).sum()) if ok.any() else float("nan")
    p_omnibus = float((1 + (np.abs(flipped[:, ok]).sum(1) >= omnibus - 1e-12).sum()) / (flips + 1)) if ok.any() else float("nan")
    return est, p, omnibus, p_omnibus


def two_sample(D: np.ndarray, fav: np.ndarray, flips: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, float]:
    """For `quality_batch`: D holds the favoured batches' values and the other batches' values
    negated (`fav` marks the favoured rows); the estimate is the mean difference, and the
    p-value permutes which batches are which, an approximation since batches share questions
    across rounds."""
    A, B = np.where(fav[:, None], D, np.nan), np.where(~fav[:, None], -D, np.nan)
    est = nanmean0(A, 0) - nanmean0(B, 0)
    values = np.where(fav[:, None], D, -D)           # every row on its own scale
    diffs = np.empty((flips, D.shape[1]))
    for i in range(flips):
        perm = rng.permutation(len(D))
        f = fav[perm]
        diffs[i] = nanmean0(np.where(f[:, None], values, np.nan), 0) - nanmean0(np.where(~f[:, None], values, np.nan), 0)
    p = (1 + (np.abs(diffs) >= np.abs(est) - 1e-12).sum(0)) / (flips + 1)
    ok = np.isfinite(est)
    omnibus = float(np.abs(est[ok]).sum()) if ok.any() else float("nan")
    p_omnibus = float((1 + (np.abs(diffs[:, ok]).sum(1) >= omnibus - 1e-12).sum()) / (flips + 1)) if ok.any() else float("nan")
    return est, p, omnibus, p_omnibus


def bootstrap_interval(D: np.ndarray, unit: np.ndarray, boot: int, rng: np.random.Generator) -> np.ndarray:
    """95% percentile intervals of each column's mean when the units are resampled with replacement: (2, features)."""
    keep = unit >= 0
    D, unit = D[keep], unit[keep]
    codes, units = pd.factorize(unit)
    valid = np.isfinite(D)
    D0 = np.where(valid, D, 0.0)
    counts_by_unit = np.zeros((len(units), D.shape[1]))
    sums_by_unit = np.zeros((len(units), D.shape[1]))
    np.add.at(counts_by_unit, codes, valid)
    np.add.at(sums_by_unit, codes, D0)
    means = np.empty((boot, D.shape[1]))
    for i in range(boot):
        w = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)).astype(float)
        n = w @ counts_by_unit
        means[i] = np.where(n > 0, (w @ sums_by_unit) / np.maximum(n, 1), np.nan)
    return np.nanpercentile(means, [2.5, 97.5], axis=0)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Which p-values are rejected at FDR q (NaN never)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    rejected = np.zeros(len(p), bool)
    if not ok.any():
        return rejected
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    thresholds = q * np.arange(1, len(order) + 1) / len(order)
    below = p[order] <= thresholds
    k = below.nonzero()[0].max() + 1 if below.any() else 0
    rejected[order[:k]] = True
    return rejected


def holm(p: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down rejections for a small family of p-values."""
    items = sorted((v, k) for k, v in p.items() if np.isfinite(v))
    out = {k: False for k in p}
    for i, (v, k) in enumerate(items):
        if v <= alpha / (len(items) - i):
            out[k] = True
        else:
            break
    return out


def analyse(plan: pd.DataFrame, calls: pd.DataFrame, features: pd.DataFrame, attributions: pd.DataFrame, mapping: pd.DataFrame,
            params: ContrastParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every test on every canonical feature, both sides. Returns the feature-level table
    (side, test, feature, calls naming it, estimate, interval, p, q, rejected at FDR 0.05) and
    the omnibus table (side, test, statistic, p, and for the primary family the Holm decision)."""
    layout = Layout(plan)
    rng = np.random.default_rng(params.seed)
    ids, Zb, Zq = outcomes(layout, calls, features, attributions, mapping, params.min_calls)
    named_calls = features.merge(mapping.dropna(subset=["canonical"])[KEY + ["canonical"]], on=KEY).drop_duplicates(KEY + ["canonical"]).groupby("canonical").size()
    rows, omni = [], []
    for side in SIDES:
        for test, (D, unit) in differences(layout, Zb, Zq, side).items():
            if test == "quality_batch":
                fav = np.zeros(len(D), bool)
                n_fav = (layout.batch_kind[unit] == CONDITIONS[SIDES[side]["homogeneous_favoured"]][0]).sum()
                fav[:n_fav] = True
                est, p, stat, p_stat = two_sample(D, fav, params.flips, rng)
                lo = hi = np.full(len(ids), np.nan)
            else:
                est, p, stat, p_stat = sign_flip(D, unit, params.flips, rng)
                lo, hi = bootstrap_interval(D, unit, params.boot, rng) if params.boot else (np.full(len(ids), np.nan),) * 2
            rejected = benjamini_hochberg(p)
            for i, f in enumerate(ids):
                rows.append({"side": side, "test": test, "feature": f, "calls": int(named_calls.get(f, 0)), "estimate": est[i], "ci_low": lo[i], "ci_high": hi[i],
                             "p": p[i], "fdr_rejected": bool(rejected[i])})
            omni.append({"side": side, "test": test, "features": len(ids), "l1_statistic": stat, "p": p_stat})
    table = pd.DataFrame(rows)
    omnibus = pd.DataFrame(omni)
    if len(table):
        table["q"] = np.nan
        for (side, test), g in table.groupby(["side", "test"]):
            table.loc[g.index, "q"] = q_values(g.p.to_numpy())
    primary = {f"{r.side}:{r.test}": r.p for r in omnibus.itertuples() if r.test in PRIMARY}
    decisions = holm(primary)
    omnibus["primary"] = omnibus.apply(lambda r: r.test in PRIMARY, axis=1)
    omnibus["holm_rejected"] = omnibus.apply(lambda r: decisions.get(f"{r.side}:{r.test}", False) if r.test in PRIMARY else None, axis=1)
    return table, omnibus


def q_values(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def raw_mapping(features: pd.DataFrame) -> pd.DataFrame:
    """A smoke-test mapping: the raw name, lower-cased and stripped of punctuation."""
    canonical = features.name.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return features[KEY].assign(canonical=canonical)
