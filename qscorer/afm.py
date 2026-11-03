"""The practice-slope model of Stage 0.1, fitted with leapfit.

    logit P(correct) = student_i + intercept_q + jump_q * [practice] + (mu + b_q) * log(1 + T)
    b_q ~ Normal(0, tau^2)

T is student i's count of earlier practice attempts on question q, so the gain from attempt 1
to attempt k is slope_q * log k, with slope_q = mu + b_q. [practice] is 0 on a baseline
(pretest) row and 1 on a practice row: `intercept_q` is the baseline level and `jump_q` the
change from it to the first practice attempt. In leapfit this is the AFM design with one
component per question on its practice rows and a second, `<q>|baseline`, on its pretest rows
(an intercept and no slope; the jump is the difference of the two intercepts). The slope block
is ridge-penalised at 1/tau^2, which is the Gaussian prior on the deviations b_q, and one
unpenalised column carries mu, so the solver returns the posterior mode of every deviation.
tau^2 is estimated by the outer loop: the EM update of a variance component under the Laplace
approximation, tau^2 <- mean over questions of (b_q^2 + posterior variance of b_q), iterated to
convergence. Every question's slope posterior is normal with mean mu + b_q and the variance of
that sum from the inverse penalised information; the same inverse gives the posterior sd of any
difference of slopes, which Stage 0.2's separation test needs.

The schedule-adjusted variant (`schedule` given: one label per row, the practice schedule of
the row's student on its question) adds kappa_c * [practice] and replaces mu with mu_c:

    logit P(correct) = student_i + intercept_q + jump_q * [practice] + kappa_c * [practice] + (mu_c + b_q) * log(1 + T)

so b_q is the question's slope relative to the average curve of the schedules it was practiced
under. One kappa is dropped (the most frequent schedule is the reference), since the five
together equal the practice indicator, which the intercepts already span. The reported slope is
then sum_c w_c mu_c + b_q, with w_c the share of slope-carrying rows under schedule c, which
keeps it on the scale of the unadjusted fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd
from leapfit.afm import AFMFit
from leapfit.data import StepData
from leapfit.design import Block, Design
from leapfit.fit import fit_logistic
from scipy import sparse
from scipy.sparse.linalg import splu

from qscorer.data import BASELINE, is_baseline

SCHEDULE_JUMP, SCHEDULE_SLOPE = "schedule_jump", "schedule_slope"


def practice_curve(data: StepData) -> list[tuple[float, ...]]:
    """log(1 + T) on practice rows; 0 on baseline rows, whose component has no slope."""
    return [tuple(0.0 if is_baseline(k) else float(np.log1p(t)) for k, t in zip(kcs, opps))
            for kcs, opps in zip(data.kcs, data.opportunities)]


def build_design(data: StepData, tau2: float, schedule: Sequence[str] | None = None) -> Design:
    """The AFM blocks (student, KC intercept, KC slope on the curve) with the slope block
    penalised at 1/tau^2, identified, then the mean-slope column(s) appended: `slope_mean:mu`,
    or with a schedule one `schedule_slope` column per schedule. Those columns are spanned by the
    deviation columns, so they must not go through `identify`; the ridge keeps the penalised
    problem strictly convex. With a schedule the jump block (one indicator per non-reference
    schedule on practice rows) goes through `identify` with the rest."""
    values = practice_curve(data)
    blocks = [
        Block.from_levels("student", [(s,) for s in data.students]),
        Block.from_levels("kc_intercept", data.kcs),
        Block.from_levels("kc_slope", data.kcs, values=values, l2=1.0 / tau2),
    ]
    if schedule is None:
        mean = Block.build("slope_mean", sparse.csr_matrix(np.array([[sum(v)] for v in values])), ["mu"])
        return Design(tuple(blocks)).identify().with_blocks(mean)
    schedule = list(schedule)
    if len(schedule) != len(values):
        raise ValueError(f"schedule has {len(schedule)} labels for {len(values)} rows")
    practice = [not all(is_baseline(k) for k in kcs) for kcs in data.kcs]
    reference = pd.Series([s for s, p in zip(schedule, practice) if p]).mode().iloc[0]
    blocks.append(Block.from_levels(SCHEDULE_JUMP, [(s,) if p and s != reference else () for s, p in zip(schedule, practice)]))
    slope = Block.from_levels(SCHEDULE_SLOPE, [(s,) if sum(v) > 0 else () for s, v in zip(schedule, values)],
                              values=[(sum(v),) if sum(v) > 0 else () for v in values])
    return Design(tuple(blocks)).identify().with_blocks(slope)


def mean_slope_columns(design: Design, schedule: Sequence[str] | None, values: list[tuple[float, ...]]) -> dict[int, float]:
    """The columns whose weighted sum is the mean slope: `mu` alone, or each schedule's slope
    weighted by its share of the rows that carry slope information (T >= 1)."""
    cols = design.columns
    if schedule is None:
        return {cols.index("slope_mean:mu"): 1.0}
    share = pd.Series([s for s, v in zip(schedule, values) if sum(v) > 0]).value_counts(normalize=True)
    return {cols.index(f"{SCHEDULE_SLOPE}:{s}"): float(w) for s, w in share.items()}


def with_tau2(design: Design, tau2: float) -> Design:
    """The same design with the slope ridge set to 1/tau^2."""
    blocks = tuple(Block(b.name, b.matrix, b.columns, np.full(len(b.columns), 1.0 / tau2), b.lower, b.upper)
                   if b.name == "kc_slope" else b for b in design.blocks)
    return Design(blocks, design.aliased)


def information(design: Design, fit) -> sparse.csc_matrix:
    """The penalised observed information X'WX + diag(l2), W = p(1-p), at the fitted solution,
    with a tiny ridge so that a separated column does not make the LU fail."""
    X = design.matrix.tocsr()
    p = fit.predict_proba(design)
    info = (X.T @ sparse.diags(p * (1 - p)) @ X + sparse.diags(design.l2)).tocsc()
    return info + sparse.identity(info.shape[0], format="csc") * 1e-10


def standard_errors(info: sparse.csc_matrix, targets: list[dict[int, float]]) -> np.ndarray:
    """Standard errors of linear combinations of coefficients from the inverse information. Each
    target is {column index: weight}: `{i: 1}` for coefficient i, `{i: 1, j: -1}` for a
    difference. Each is one solve through a sparse LU, so no dense inverse is formed."""
    variances = np.full(len(targets), np.nan)
    try:
        lu = splu(info)
        for start in range(0, len(targets), 512):
            chunk = targets[start:start + 512]
            rhs = np.zeros((info.shape[0], len(chunk)))
            for k, target in enumerate(chunk):
                for i, weight in target.items():
                    rhs[i, k] += weight
            sol = lu.solve(rhs)
            variances[start:start + len(chunk)] = [(rhs[:, k] * sol[:, k]).sum() for k in range(len(chunk))]
    except RuntimeError:
        cov = np.linalg.pinv(info.toarray())
        variances = [sum(a * b * cov[i, j] for i, a in t.items() for j, b in t.items()) for t in targets]
    return np.sqrt(np.clip(np.asarray(variances, float), 0, None))


@dataclass
class SlopeFit:
    fit: AFMFit
    design: Design
    mean_columns: dict[int, float]      # columns and weights whose sum is the mean slope mu
    tau2: float
    iterations: list[dict]              # tau2 per EM iteration
    laplace_log_marginal: float
    schedule_terms: dict[str, float] | None = None  # `jump:<schedule>` and `slope:<schedule>` when adjusted

    @property
    def mu(self) -> float:
        return float(sum(w * self.fit.weights[i] for i, w in self.mean_columns.items()))

    @cached_property
    def slope_columns(self) -> dict[str, int]:
        """Question id -> column of its slope deviation b_q (questions with one opportunity have none)."""
        return {c.split(":", 1)[1]: i for i, c in enumerate(self.design.columns) if c.startswith("kc_slope:")}

    @cached_property
    def info(self) -> sparse.csc_matrix:
        return information(self.design, self.fit)

    def slopes(self) -> pd.DataFrame:
        """Per question: the posterior mean `slope` (mu + b_q), its sd, and whether the question
        is separated (an intercept or slope with no finite estimate)."""
        cols = list(self.slope_columns.values())
        separated = set(self.fit.separated.in_blocks("kc_slope", "kc_intercept"))
        return pd.DataFrame({"slope": self.mu + self.fit.weights[cols],
                             "slope_sd": standard_errors(self.info, [{**self.mean_columns, i: 1.0} for i in cols]),
                             "separated": [q in separated for q in self.slope_columns]},
                            index=pd.Index(list(self.slope_columns), name="id"))

    def jumps(self) -> pd.DataFrame:
        """Per question with both components: practice intercept minus baseline intercept, with
        its sd (under the adjusted fit, at the reference schedule)."""
        col = {c: i for i, c in enumerate(self.design.columns)}
        pairs = [(c.split(":", 1)[1], i, col[c + BASELINE]) for c, i in col.items()
                 if c.startswith("kc_intercept:") and not is_baseline(c) and c + BASELINE in col]
        w = self.fit.weights
        return pd.DataFrame({"jump": [w[i] - w[j] for _, i, j in pairs],
                             "jump_sd": standard_errors(self.info, [{i: 1.0, j: -1.0} for _, i, j in pairs])},
                            index=pd.Index([q for q, _, _ in pairs], name="id"))

    def difference_sd(self, high: list[str], low: list[str]) -> np.ndarray:
        """The posterior sd of slope_h - slope_l for each pair, from the joint posterior."""
        idx = self.slope_columns
        return standard_errors(self.info, [{idx[h]: 1.0, idx[l]: -1.0} for h, l in zip(high, low)])

    def kc_values(self, data: StepData) -> pd.DataFrame:
        """leapfit's per-component table: intercepts for the average student, separation."""
        return self.fit.kc_values(data)

    def student_effects(self, data: StepData) -> pd.Series:
        """Every student's ability at the sum-to-zero point, so that `effect + intercept` is the
        student's logit on a question."""
        theta, _ = self.fit.centred_students(data)
        return theta


def fit_slopes(data: StepData, schedule: Sequence[str] | None = None, tau2: float = 0.05,
               max_iter: int = 50, tol: float = 1e-3) -> SlopeFit:
    """Fit the model, schedule-adjusted when `schedule` (one label per row) is given. `tau2` is
    the starting value of the EM loop; `tol` its relative tolerance."""
    design = build_design(data, tau2, schedule)
    cols = design.columns
    idx = [i for i, c in enumerate(cols) if c.startswith("kc_slope:")]
    mean_columns = mean_slope_columns(design, schedule, practice_curve(data))
    label = "slopes-schedule" if schedule is not None else "slopes"
    w0, iterations = None, []
    for it in range(max_iter):
        design = with_tau2(design, tau2)
        fit = fit_logistic(design, data.y, w0=w0, warn_separated=False, label=label, result_type=AFMFit)
        w0 = fit.weights
        b = fit.weights[idx]
        v = standard_errors(information(design, fit), [{i: 1.0} for i in idx]) ** 2
        tau2_new = float(np.mean(b * b + v))
        iterations.append({"iteration": it, "tau2": tau2, "tau2_next": tau2_new,
                           "mu": float(sum(w * fit.weights[i] for i, w in mean_columns.items())),
                           "log_likelihood": fit.ll_unpenalized, "optimal": fit.is_optimal})
        converged = abs(tau2_new - tau2) <= tol * tau2
        tau2 = tau2_new
        if converged:
            break
    design = with_tau2(design, tau2)
    fit = fit_logistic(design, data.y, w0=w0, warn_separated=False, label=label, result_type=AFMFit)
    terms = ({c.replace("schedule_", ""): float(fit.weights[i]) for i, c in enumerate(cols) if c.startswith("schedule_")}
             if schedule is not None else None)
    return SlopeFit(fit, design, mean_columns, tau2, iterations, laplace_log_marginal(fit, design, len(idx), tau2), terms)


def laplace_log_marginal(fit, design: Design, n_deviations: int, tau2: float) -> float:
    """log p(y | tau^2) under the Laplace approximation, with the deviations *and* the fixed
    effects integrated out under flat priors: the penalised log-likelihood at the joint mode,
    minus Q/2 log tau^2, minus half the log determinant of the full penalised information. Up to
    a constant this is the REML-type criterion whose maximum the EM loop converges to."""
    lu = splu(information(design, fit))
    logdet = float(np.log(np.abs(lu.U.diagonal())).sum())
    return float(fit.ll - 0.5 * n_deviations * np.log(tau2) - 0.5 * logdet)
