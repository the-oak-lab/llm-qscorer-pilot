"""Stage 2.4 (`power`): a simulation of feature discovery on a batch plan.

Simulates Gemini's question-level attributions of one feature on the planned batches under a
simple generative model, applies the planned tests, and reports how often each rejects: the
power at a range of effect sizes, and the false-positive rate when there is nothing to find.
One feature of good polarity is simulated; the bad-feature side (E2, E4, E6, E8, E10) is its
mirror image and has the same operating characteristics.

Model. Question j in batch b under condition c is attributed the feature with

    logit P(Z = 1) = alpha + delta [j high-learning] + kappa [j labelled HIGH in c] + gamma [b mixed] + u_j + v_b + w_bc,

where alpha gives the base attribution rate, `delta` is the quality effect (the feature really
is more common in high-learning questions), `kappa` the label effect (Gemini follows the label
it is shown: the high members in E7, the low members in E9), `gamma` a batch-composition
effect, u_j ~ N(0, sigma_question^2) the question's stable tendency, v_b ~ N(0, sigma_batch^2)
the batch's, and w_bc ~ N(0, sigma_call^2) the call's. With `samples` calls per batch and
condition, Z is the mean over the calls.

Tests. Each statistic is a mean of within-pair differences and its p-value comes from a
sign-flip permutation test (two-sided). Signs are flipped by the unit that is exchangeable
under the null: when the two conditions compared are two calls on the same mixed batch, the
batch (both of its pairs flip together, since the calls share the batch's effect); when the
comparison is between the two members of a pair, the pair. Two of the tests have no exact
flipping unit, because pairs whose members share a homogeneous batch are dependent through
that batch's effect; they flip by pair and the simulation reports how far that is from nominal.

- quality (E1 vs E3, the design's primary): the high member's attribution in its good batches
  minus the low member's in its bad batches, each averaged over rounds. Flips by pair: approximate.
- quality_mixed (within E5): high minus low member in the same unlabelled mixed batch and
  call. Flips by pair: exact, since under the null the two members' roles are exchangeable.
- inversion (E7 vs E9, primary): the within-pair difference under true labels minus the same
  under inverted labels, on the same batch. Flips by batch: exact.
- labels (E7 vs E5, secondary): the within-pair difference under true labels minus under no
  labels. Flips by batch: exact.
- composition (E5 vs E1/E3, secondary): the pair's mean attribution in its mixed batch minus
  the mean of the members' attributions in their homogeneous batches of the same round. Flips
  by pair: approximate.

Power is reported at alpha = 0.05 and at 0.0125, the strictest Holm step for the design's
family of four primary tests (quality and inversion, each for good and bad features). An
exploratory family of `features` independent features, `enriched` of them with the quality
effect, gives the expected true and false discoveries under Benjamini-Hochberg at q = 0.05,
built on the within-E5 quality test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.special import expit, logit

from qscorer.batches import CONDITIONS

TESTS = ("quality", "quality_mixed", "inversion", "labels", "composition")


class PowerParams(BaseModel):
    base_rate: float = Field(0.25, gt=0, lt=1, description="attribution rate of the feature to a low-learning question in an unlabelled batch")
    sigma_question: float = Field(1.0, ge=0, description="sd of the question effect (logits)")
    sigma_batch: float = Field(0.5, ge=0, description="sd of the batch effect (logits)")
    sigma_call: float = Field(0.5, ge=0, description="sd of the call effect (logits)")
    deltas: list[float] = Field([0.0, 0.5, 1.0, 1.5, 2.0], description="quality effects tried (logits), with kappa = 0")
    kappas: list[float] = Field([0.5, 1.0, 1.5, 2.0], description="label effects tried (logits), with delta = 0")
    gamma: float = Field(0.0, description="batch-composition effect (logits)")
    samples: int = Field(1, ge=1, description="calls per batch and condition")
    sims: int = Field(500, ge=1)
    flips: int = Field(999, ge=1, description="sign flips per permutation test")
    features: int = Field(20, ge=1, description="size of the exploratory feature family")
    enriched: int = Field(5, ge=0, description="features in the family that carry the quality effect")
    seed: int = 0


class Layout:
    """The plan as index arrays: for each kind and pool, the plan row of each pair's member in
    each round (-1 when the pair sits out), plus each row's question and batch index."""

    def __init__(self, plan: pd.DataFrame):
        self.plan = plan.reset_index(drop=True)
        self.pairs = np.sort(self.plan.pair.unique())
        self.rounds = np.sort(self.plan["round"].unique())
        self.question = pd.factorize(self.plan.id)[0]
        self.batch = pd.factorize(self.plan.batch)[0]
        self.n_questions, self.n_batches = self.question.max() + 1, self.batch.max() + 1
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

    def pick(self, z: np.ndarray, kind: str, pool: str) -> np.ndarray:
        """z at each pair's member in each round, NaN where the pair sits out: shape (rounds, pairs)."""
        slot = self.slot[kind, pool]
        return np.where(slot >= 0, z[np.maximum(slot, 0)], np.nan)


def simulate(layout: Layout, params: PowerParams, delta: float, kappa: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One draw of the attributions, per condition, over the plan rows."""
    alpha = logit(params.base_rate)
    u = rng.normal(0, params.sigma_question, layout.n_questions)[layout.question]
    v = rng.normal(0, params.sigma_batch, layout.n_batches)[layout.batch]
    base = alpha + delta * layout.high + params.gamma * layout.mixed + u + v
    z = {}
    for c, (_, polarity, labels) in CONDITIONS.items():
        if polarity != "good":
            continue
        label = (layout.high if labels == "true" else ~layout.high if labels == "inverted" else np.zeros(len(base), bool))
        p = expit(base + kappa * label)
        draws = np.empty((params.samples, len(base)))
        for k in range(params.samples):
            w = rng.normal(0, params.sigma_call, layout.n_batches)[layout.batch]
            draws[k] = rng.random(len(base)) < expit(logit(p) + w)
        z[c] = draws.mean(0)
    return z


def differences(layout: Layout, z: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per test: the within-pair differences and, for each, the unit whose sign flips together
    (a pair, or a mixed batch in a round). Per-round differences are NaN where a pair sits out."""
    rounds, pairs = layout.slot["mixed", "high"].shape
    pair_unit = np.broadcast_to(np.arange(pairs), (rounds, pairs))
    slot = layout.slot["mixed", "high"]
    batch_unit = np.where(slot >= 0, layout.batch[np.maximum(slot, 0)], -1)

    def avg(c, kind, pool):
        v = layout.pick(z[c], kind, pool)
        n = np.isfinite(v).sum(0)
        return np.where(n > 0, np.nansum(v, 0) / np.maximum(n, 1), np.nan)
    m = lambda c, pool: layout.pick(z[c], "mixed", pool)
    within = lambda c: m(c, "high") - m(c, "low")
    return {
        "quality": (avg("E1", "good", "high") - avg("E3", "bad", "low"), np.arange(pairs)),
        "quality_mixed": (within("E5"), pair_unit),
        "inversion": (within("E7") - within("E9"), batch_unit),
        "labels": (within("E7") - within("E5"), batch_unit),
        "composition": ((m("E5", "high") + m("E5", "low")) / 2 - (layout.pick(z["E1"], "good", "high") + layout.pick(z["E3"], "bad", "low")) / 2, pair_unit),
    }


def sign_flip_p(d: np.ndarray, unit: np.ndarray, flips: int, rng: np.random.Generator) -> float:
    """Two-sided p-value of the mean of `d` when the signs of its entries are flipped by `unit`
    (entries with the same unit flip together); NaN entries are left out."""
    keep = np.isfinite(d)
    d, unit = d[keep], np.asarray(unit)[keep]
    codes, units = pd.factorize(unit)
    t = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(flips, len(units)))
    return float((1 + (np.abs(signs[:, codes] @ d) / len(d) >= t - 1e-12).sum()) / (flips + 1))


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Which p-values are rejected at FDR q."""
    order = np.argsort(p)
    thresholds = q * (np.arange(1, len(p) + 1)) / len(p)
    below = p[order] <= thresholds
    k = below.nonzero()[0].max() + 1 if below.any() else 0
    rejected = np.zeros(len(p), bool)
    rejected[order[:k]] = True
    return rejected


def run_power(plan: pd.DataFrame, params: PowerParams, progress=print) -> pd.DataFrame:
    """One row per scenario: the implied attribution rates and the rejection rate of each test
    at alpha 0.05 and 0.0125. The delta scenarios also carry the exploratory family's expected
    true and false discoveries, built from the scenario's own quality p-values for the enriched
    features and the null scenario's for the rest (the features are independent, so this is the
    family's distribution without simulating it feature by feature)."""
    layout = Layout(plan)
    rng = np.random.default_rng(params.seed)
    scenarios = [(d, 0.0) for d in params.deltas] + [(0.0, k) for k in params.kappas]
    rows, quality_p = [], {}
    for delta, kappa in scenarios:
        p = {t: np.empty(params.sims) for t in TESTS}
        for i in range(params.sims):
            z = simulate(layout, params, delta, kappa, rng)
            for t, (d, unit) in differences(layout, z).items():
                p[t][i] = sign_flip_p(d, unit, params.flips, rng)
        if kappa == 0.0:
            quality_p[delta] = p["quality_mixed"]
        row = {"delta": delta, "kappa": kappa, "rate_low": params.base_rate, "rate_high": float(expit(logit(params.base_rate) + delta)),
               "rate_labelled_high": float(expit(logit(params.base_rate) + kappa))}
        for t in TESTS:
            row[f"{t}_0.05"] = float((p[t] <= 0.05).mean())
            row[f"{t}_0.0125"] = float((p[t] <= 0.0125).mean())
        rows.append(row)
        progress(f"  delta {delta:.1f} kappa {kappa:.1f}: " + ", ".join(f"{t} {row[f'{t}_0.05']:.2f}" for t in TESTS))
    if 0.0 in quality_p:
        for row in rows:
            if row["kappa"] == 0.0:
                found_true, found_false = [], []
                for _ in range(params.sims):
                    family = np.concatenate([rng.choice(quality_p[row["delta"]], params.enriched), rng.choice(quality_p[0.0], params.features - params.enriched)])
                    rejected = benjamini_hochberg(family, 0.05)
                    found_true.append(rejected[:params.enriched].sum())
                    found_false.append(rejected[params.enriched:].sum())
                row["family_true_discoveries"], row["family_false_discoveries"] = float(np.mean(found_true)), float(np.mean(found_false))
    return pd.DataFrame(rows)
