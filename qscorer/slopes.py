"""Stage 0.1 (`slopes`): estimate every question's practice slope from student data.

Fit the model in `afm.py` to the first attempts of the practice phase and the baseline phase,
and write one row per question: its item set, the counts behind it, the fitted baseline and
first-practice levels, the jump, the slope posterior (mean, sd, P(slope > 0), the implied change
in accuracy over the practice window) and the observed accuracies that Stage 0.2 matches on. A
question gets no posterior when it is separated or fewer than `min_support` students have a
second practice opportunity. Split-half reliability (random halves of the students, refitted)
says how much of the between-question spread in slopes is signal.
"""
from __future__ import annotations

import datetime as dt
import platform
from dataclasses import dataclass
from importlib import metadata

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

from qscorer.afm import SlopeFit, fit_slopes
from qscorer.config import DatasetConfig, SlopeParams
from qscorer.data import BASELINE, KC_MODEL, is_baseline, item_sets, load_questions, observed_accuracy, practice_steps, read_datashop, to_step_data

COUNTS = ["n_students", "n_observations", "n_second_opportunity", "n_baseline", "n_pretest", "n_first_practice"]


@dataclass
class SlopeResult:
    table: pd.DataFrame      # one row per question in the bank
    summary: dict
    log: dict
    students: pd.DataFrame   # per student: the fitted ability (logit)


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def question_table(steps: pd.DataFrame, config: DatasetConfig, params: SlopeParams) -> tuple[pd.DataFrame, SlopeFit, pd.DataFrame]:
    """Fit the model to `steps` and return one row per question with data, the fit, and the
    students' abilities."""
    data = to_step_data(steps)
    sf = fit_slopes(data)
    practice, baseline = steps[steps["Role"].eq("practice")], steps[steps["Role"].eq("baseline")]
    opp = practice[f"Opportunity ({KC_MODEL})"].astype(int)
    g = practice.groupby(config.step_column)
    t = pd.DataFrame({
        "n_students": g[config.student_column].nunique(),
        "n_observations": g.size(),
        "n_second_opportunity": practice[opp >= 2].groupby(config.step_column)[config.student_column].nunique(),
        "n_baseline": baseline.groupby(config.step_column)[config.student_column].nunique(),
    }).fillna(0).astype(int)
    kv = sf.kc_values(data).set_index("KC Name")
    kc, base = kv[~kv.index.map(is_baseline)], kv[kv.index.map(is_baseline)].copy()
    base.index = base.index.str.replace(BASELINE, "", regex=False)
    t["baseline_logit"] = base["Intercept (logit)"].reindex(t.index)
    t["p_baseline"] = expit(t["baseline_logit"])
    t["intercept_logit"] = kc["Intercept (logit)"].reindex(t.index)
    t["p_opportunity_1"] = kc["Intercept (probability) at Opportunity 1"].reindex(t.index)
    jumps = sf.jumps().reindex(t.index)
    base_separated = base["Separated"].reindex(t.index).fillna(False).astype(bool)
    ok_jump = jumps.jump.notna() & ~base_separated & (t.n_baseline >= params.min_support)
    t["jump"] = jumps.jump.where(ok_jump)
    t["jump_sd"] = jumps.jump_sd.where(ok_jump)
    t["p_jump_positive"] = norm.sf(-t["jump"] / t["jump_sd"])
    slopes = sf.slopes().reindex(t.index)
    t["supported"] = slopes.slope.notna() & ~slopes.separated.fillna(True).astype(bool) & (t.n_second_opportunity >= params.min_support)
    t["slope"] = slopes.slope.where(t.supported)
    t["slope_sd"] = slopes.slope_sd.where(t.supported)
    t["p_positive"] = norm.sf(-t["slope"] / t["slope_sd"])
    window = params.max_opportunities or int(opp.max())
    t["p_change_over_window"] = expit(t["intercept_logit"] + t["slope"] * np.log(window)) - expit(t["intercept_logit"])
    t = t.join(observed_accuracy(steps, config, list(t.index)))
    t.index.name = "id"
    theta = sf.student_effects(data)
    return t, sf, pd.DataFrame({"student": theta.index, "effect": theta.to_numpy()})


def split_half_reliability(steps: pd.DataFrame, config: DatasetConfig, params: SlopeParams) -> dict:
    """Fit the model on two random halves of the students, several times, and report how well
    the slopes (and jumps) of the two halves agree, Spearman-Brown corrected to the full sample."""
    students = np.array(sorted(steps[config.student_column].unique()))
    rows = []
    for k in range(params.reliability_splits):
        rng = np.random.default_rng(params.seed + k)
        half = set(rng.permutation(students)[: len(students) // 2])
        parts = [steps[steps[config.student_column].isin(half)], steps[~steps[config.student_column].isin(half)]]
        a, b = (question_table(part, config, params)[0] for part in parts)
        both = a.join(b, suffixes=("_a", "_b"), how="inner")
        ok = both.slope_a.notna() & both.slope_b.notna()
        okj = both.jump_a.notna() & both.jump_b.notna()
        r = float(pearsonr(both.slope_a[ok], both.slope_b[ok])[0]) if ok.sum() > 2 else float("nan")
        r_jump = float(pearsonr(both.jump_a[okj], both.jump_b[okj])[0]) if okj.sum() > 2 else float("nan")
        rows.append({"split": k, "questions": int(ok.sum()), "slope_pearson_r": r,
                     "slope_spearman_rho": float(spearmanr(both.slope_a[ok], both.slope_b[ok])[0]) if ok.sum() > 2 else float("nan"),
                     "slope_spearman_brown": spearman_brown(r), "jump_spearman_brown": spearman_brown(r_jump)})
    splits = pd.DataFrame(rows)
    return {"splits": rows, "median": {c: float(splits[c].median()) for c in splits.columns if c != "split"}}


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")


def run_slopes(config: DatasetConfig, params: SlopeParams) -> SlopeResult:
    """Stage 0.1 end to end: load, fit, tabulate, check reliability, and summarise."""
    questions = load_questions(config.questions)
    ids = list(questions)
    tx = read_datashop(config.transactions)
    steps, log = practice_steps(tx, config, params, set(ids))
    log["questions_in_file"] = len(ids)
    table, sf, students = question_table(steps, config, params)
    table = table.reindex(ids)  # every question in the bank, in file order
    table.insert(0, "set", pd.Series(item_sets(tx, config, ids)))
    for c in COUNTS:
        table[c] = table[c].fillna(0).astype(int)
    table["supported"] = table["supported"].fillna(False).astype(bool)
    reliability = split_half_reliability(steps, config, params) if params.reliability_splits else None
    s = table[table.supported]
    summary = {
        "dataset": config.model_dump(mode="json"),
        "params": params.model_dump(mode="json"),
        "fit": {"observations": sf.fit.n_obs, "parameters": sf.fit.n_params, "log_likelihood": sf.fit.ll_unpenalized,
                "optimal": sf.fit.is_optimal, "max_free_gradient": sf.fit.max_free_gradient, "n_iter": sf.fit.n_iter},
        "prior": {"mu": sf.mu, "tau": float(np.sqrt(sf.tau2)), "em_iterations": len(sf.iterations),
                  "tau2_path": [round(i["tau2"], 6) for i in sf.iterations] + [round(sf.tau2, 6)],
                  "laplace_log_marginal": sf.laplace_log_marginal},
        "questions": {"in_file": len(ids), "with_data": int(table.n_students.gt(0).sum()), "supported": int(table.supported.sum()),
                      "sets": table.set.value_counts().sort_index().to_dict()},
        "slopes": {"median": float(s.slope.median()), "q10": float(s.slope.quantile(.1)), "q90": float(s.slope.quantile(.9)),
                   "median_sd": float(s.slope_sd.median()), "share_positive": float((s.slope > 0).mean()),
                   "share_p_positive_at_least_0.95": float((s.p_positive >= .95).mean()),
                   "share_p_positive_at_most_0.70": float((s.p_positive <= .70).mean())} if len(s) else {},
        "jump": {"questions": int(s.jump.notna().sum()), "median": float(s.jump.median()), "median_sd": float(s.jump_sd.median()),
                 "share_positive": float((s.jump > 0).mean()), "share_p_positive_at_least_0.95": float((s.p_jump_positive >= .95).mean())}
                if s.jump.notna().any() else None,
        "students": {"n": int(len(students)), "sd_of_effects": float(students.effect.std(ddof=1))},
        "reliability": reliability,
        "versions": {"python": platform.python_version(), **{p: metadata.version(p) for p in ("leapfit", "pandas", "numpy", "scipy")}},
        "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return SlopeResult(table=table.reset_index(), summary=summary, log=log, students=students)
