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

Tests: those of `qscorer/contrasts.py`, run through the same code as the real analysis. The
batch-level outcome (the feature named for the batch) is taken as "attributed to any of its
questions". The exact tests flip by batch (inversion, labels, inversion_question) or by pair
(quality_mixed); quality and composition flip by pair although pairs sharing a homogeneous
batch are dependent, and the simulation reports how far that is from nominal.

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
from qscorer.contrasts import Layout, benjamini_hochberg, differences, sign_flip

TESTS = ("quality", "quality_mixed", "inversion", "inversion_question", "labels", "composition")


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
        conds = list(CONDITIONS)
        for i in range(params.sims):
            z = simulate(layout, params, delta, kappa, rng)
            Zq = np.full((1, len(conds), len(layout.plan)), np.nan)
            for c, v in z.items():
                Zq[0, conds.index(c)] = v
            Zb = np.full((1, len(conds), layout.n_batches), np.nan)
            for c in z:
                named = pd.Series(Zq[0, conds.index(c)]).groupby(layout.batch).max()
                Zb[0, conds.index(c), named.index.to_numpy()] = named.to_numpy()
            for t, (D, unit) in differences(layout, Zb, Zq, "good").items():
                if t in p:
                    p[t][i] = sign_flip(D, unit, params.flips, rng)[1][0]
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
