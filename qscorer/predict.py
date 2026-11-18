"""Stage 3.3 (`predict`): a ridge regression of the practice slope on the checklist scores.

Training: the questions of one dataset with their Stage 0 posterior slopes and their Stage 3.2
scores (share of calls saying present, per item). The slope and the scores are demeaned
within item set, so the coefficients are within-set effects and the item sets' different
students and content do not enter them; observations are weighted by the precision of the
slope. The ridge penalty is chosen by leave-one-set-out cross-validation on the within-set
Spearman correlation of prediction and slope, which is what a user of the predictor wants:
the ordering of questions. The final model is refitted on every set; its intercept is the
pooled mean slope minus the coefficients applied to the pooled mean scores, so a new question
with no item set gets a slope on the pooled scale.

A new question's prediction has three parts: the predicted slope; a label, "low improvement
expected" when the prediction is below a cutoff and "improvement expected" otherwise; and its
flaws. The cutoff is a quantile (`label_quantile`, default the 25th percentile) of either the
training predictions (`fixed`, the same for every question) or the predictions of the batch
scored together (`relative`, which matches the within-set ordering the model was fitted for,
but needs a batch and moves when the batch changes). The flaws are the checklist
items present in it (scored at least `flaw_present`) that Gemini consistently claimed harm improvement. The list of
flaws does not depend on the model that scores the questions; only the score at which a flaw counts as present does
(half the calls for Gemini, a probability of 0.7 for Jev). An item is a flaw when at
least `flaw_share` of Gemini's claims about it say so (`claimed_good_share` from
`scripts/run_direction.py` at most 1 - `flaw_share`) and those claims come from at least
`flaw_min_batches` discovery batches. The flaws come from Gemini's claims, not from the coefficients, and the
two often disagree: a longer correct option is a flaw by Gemini's claims and predicts a steeper
slope. The model also carries a feature table: each item's coefficient and direction, how stable
the sign is over bootstrap refits, its prevalence in the training questions, the direction
Gemini claimed for it, and whether it is a flaw. Evaluation against a held-out dataset: Spearman
correlation overall and within sets, and the AUC of credibly high- against credibly low-learning
questions.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qscorer.config import PredictParams


@dataclass
class FitResult:
    model: dict                 # everything `apply` needs, serialisable
    evaluation: dict
    predictions: pd.DataFrame   # the training questions with their in-sample and leave-one-set-out predictions
    cv: pd.DataFrame            # one row per penalty tried


def ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted ridge without intercept (X and y are already demeaned)."""
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw + alpha * np.eye(X.shape[1]), Xw.T @ y)


def demean_within(A: np.ndarray, sets: np.ndarray) -> np.ndarray:
    out = A.astype(float).copy()
    for s in np.unique(sets):
        m = sets == s
        out[m] -= out[m].mean(axis=0)
    return out


def labels(slopes: pd.DataFrame, params: PredictParams) -> pd.Series:
    """'high', 'low' or '' per question from the posterior probability of a positive slope."""
    p = slopes.p_positive
    return pd.Series(np.where(p >= params.high_p, "high", np.where(p <= params.low_p, "low", "")), index=slopes.index)


def auc(score: np.ndarray, label: np.ndarray) -> float | None:
    hi, lo = score[label == "high"], score[label == "low"]
    if not len(hi) or not len(lo):
        return None
    return float((hi[:, None] > lo[None, :]).mean() + 0.5 * (hi[:, None] == lo[None, :]).mean())


def correlations(pred: np.ndarray, slope: np.ndarray, sets: np.ndarray, label: np.ndarray, rng: np.random.Generator | None = None, boot: int = 2000) -> dict:
    """Spearman overall (with a bootstrap interval when `rng` is given), within sets (weighted by size), and the high-vs-low AUC."""
    e = {"questions": int(len(pred)), "spearman": float(spearmanr(pred, slope)[0]) if len(pred) >= 5 else None}
    if rng is not None and len(pred) >= 5:
        bs = [spearmanr(pred[i], slope[i])[0] for i in (rng.integers(0, len(pred), len(pred)) for _ in range(boot))]
        e["spearman_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    within = {str(s): (float(spearmanr(pred[sets == s], slope[sets == s])[0]), int((sets == s).sum())) for s in np.unique(sets) if (sets == s).sum() >= 5}
    e["within_sets"] = {k: v[0] for k, v in within.items()}
    e["within_sets_weighted"] = float(sum(v[0] * v[1] for v in within.values()) / sum(v[1] for v in within.values())) if within else None
    e["auc_high_vs_low"] = auc(pred, label)
    e["high_and_low"] = [int((label == "high").sum()), int((label == "low").sum())]
    return e


def fit(scores: pd.DataFrame, slopes: pd.DataFrame, checklist: list[dict], params: PredictParams, scoring: dict, dataset: str,
        claimed: pd.DataFrame | None = None, boot: int = 500) -> FitResult:
    """`scores`: one row per question (`id`, one column per item id). `slopes`: the Stage 0 table.
    `scoring`: how the scores were produced (model, thinking, temperature, samples, context, system), kept in the model for the tool.
    `claimed`: indexed by item, `claimed_good_share` (the share of Gemini's claims that its presence helps) and `bad_batches` (the
    discovery batches with a claim that it harms), from scripts/run_direction.py; reported next to the fitted direction, and they
    decide which items are flaws. The sign of each coefficient is checked over `boot`
    bootstrap refits."""
    item_ids = [it["id"] for it in checklist]
    t = scores.set_index("id")[item_ids].join(slopes.set_index("id")[["set", "slope", "slope_sd", "p_positive"]], how="inner").dropna(subset=["slope"])
    X = t[item_ids].to_numpy(float); y = t.slope.to_numpy(float); sets = t["set"].to_numpy()
    w = 1 / np.maximum(t.slope_sd.to_numpy(float), 1e-6) ** 2; w = w / w.mean()
    lab = labels(t, params).to_numpy()
    Xc, yc = demean_within(X, sets), demean_within(y[:, None], sets)[:, 0]
    rows, held_out = [], {}
    for alpha in params.alphas:
        pred = np.full(len(y), np.nan)
        for s in np.unique(sets):
            train, test = sets != s, sets == s
            beta = ridge(demean_within(X[train], sets[train]), demean_within(y[train][:, None], sets[train])[:, 0], w[train], alpha)
            pred[test] = X[test] @ beta
        held_out[alpha] = pred
        e = correlations(pred, y, sets, lab)
        rows.append({"alpha": alpha, "cv_within_sets_spearman": e["within_sets_weighted"], "cv_auc_high_vs_low": e["auc_high_vs_low"]})
    cv = pd.DataFrame(rows)
    best = float(cv.alpha[cv.cv_within_sets_spearman.fillna(-1).idxmax()])
    beta = ridge(Xc, yc, w, best)
    intercept = float(np.average(y, weights=w) - X.mean(axis=0) @ beta)
    fitted = intercept + X @ beta
    rng = np.random.default_rng(params.seed)
    signs = np.zeros(len(item_ids))
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        b = ridge(demean_within(X[i], sets[i]), demean_within(y[i][:, None], sets[i])[:, 0], w[i], best)
        signs += np.sign(b) == np.sign(beta)
    c = claimed if claimed is not None else pd.DataFrame(columns=["claimed_good_share", "bad_batches"])
    good = {i: float(c.at[i, "claimed_good_share"]) for i in item_ids if i in c.index and not pd.isna(c.at[i, "claimed_good_share"])}
    batches = {i: int(c.at[i, "bad_batches"]) for i in item_ids if i in c.index}
    flaws = [i for i in item_ids if i in good and good[i] <= 1 - params.flaw_share and batches[i] >= params.flaw_min_batches]
    features = [{"id": it["id"], "name": it["name"], "part": it["part"], "definition": it["definition"], "criterion": it["criterion"],
                 "coefficient": float(b), "direction": "more learning" if b > 0 else "less learning", "sign_stability": float(s_ / boot),
                 "prevalence": float(X[:, k].mean()), "claimed_good_share": good.get(it["id"]), "bad_batches": batches.get(it["id"]), "flaw": it["id"] in flaws}
                for k, (it, b, s_) in enumerate(zip(checklist, beta, signs))]
    evaluation = {"training_questions": int(len(t)), "sets": {str(s): int((sets == s).sum()) for s in np.unique(sets)}, "alpha": best,
                  "cross_validated": cv[cv.alpha == best].iloc[0].to_dict(),
                  "in_sample": correlations(fitted, y, sets, lab, rng),
                  "score_prevalence": {i: float(X[:, k].mean()) for k, i in enumerate(item_ids)}}
    model = {"dataset": dataset, "fitted_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "scoring": scoring,
             "items": [{k: it[k] for k in ("id", "name", "definition", "criterion", "part")} for it in checklist],
             "intercept": intercept, "coefficients": {i: float(b) for i, b in zip(item_ids, beta)}, "alpha": best,
             "features": sorted(features, key=lambda f: -f["coefficient"]),
             "training_predictions": sorted(float(v) for v in fitted), "training_slope_quantiles": [float(np.quantile(y, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
             "label_cutoff": float(np.quantile(fitted, params.label_quantile)), "flaws": flaws,
             "params": params.model_dump(mode="json")}
    # cv_predicted_slope: each question's prediction from the model fitted without its item set; it has no intercept, so compare it within a set
    t = t.assign(cv_predicted_slope=held_out[best])
    predictions = apply(model, scores[scores.id.isin(t.index)]).merge(t[["set", "slope", "slope_sd", "p_positive", "cv_predicted_slope"]].reset_index(), on="id")
    return FitResult(model, evaluation, predictions, cv)


def label_cutoff(model: dict, pred: np.ndarray, cutoff: str = "fixed") -> float:
    """The prediction below which a question is labelled low: the model's quantile of the training predictions (`fixed`) or of `pred`,
    the batch being scored (`relative`, at least 4 questions so that the quantile is at least one question)."""
    q = model.get("params", {}).get("label_quantile", 0.25)
    if cutoff == "fixed":
        return model.get("label_cutoff", float(np.quantile(model["training_predictions"], q)))
    if cutoff == "relative":
        if len(pred) * q < 1:
            raise ValueError(f"a relative cutoff at the {q:.0%} quantile needs at least {int(np.ceil(1 / q))} questions, got {len(pred)}")
        return float(np.quantile(pred, q))
    raise ValueError(f"unknown cutoff {cutoff!r}: fixed or relative")


def apply(model: dict, scores: pd.DataFrame, present: float | None = None, cutoff: str = "fixed") -> pd.DataFrame:
    """For each row of `scores`: the predicted slope, the label (`cutoff` fixed or relative, see `label_cutoff`), and the flaws scored at
    least `present` ("; "-separated, id and name; default the model's `flaw_present`: 0.5 for Gemini, 0.7 for Jev)."""
    present = model.get("params", {}).get("flaw_present", 0.5) if present is None else present
    item_ids = [it["id"] for it in model["items"]]
    X = scores[item_ids].to_numpy(float)
    pred = model["intercept"] + X @ np.array([model["coefficients"][i] for i in item_ids])
    cut = label_cutoff(model, pred, cutoff)
    names = {it["id"]: it["name"] for it in model["items"]}
    flaws = model.get("flaws", [])
    s = scores.reset_index(drop=True)
    found = ["; ".join(f"{i} {names[i]}" for i in flaws if row[i] >= present) for _, row in s.iterrows()]
    return pd.DataFrame({"id": s.id.to_numpy(), "predicted_slope": pred,
                         "label": np.where(pred < cut, "low improvement expected", "improvement expected"), "flaws": found})


def feature_table(model: dict) -> pd.DataFrame:
    """The model's features, most positive coefficient first."""
    return pd.DataFrame(model["features"])[["id", "direction", "coefficient", "sign_stability", "prevalence", "claimed_good_share", "bad_batches", "flaw", "part", "name", "definition", "criterion"]]


def evaluate(model: dict, scores: pd.DataFrame, slopes: pd.DataFrame, params: PredictParams) -> dict:
    """The model's predictions against another dataset's Stage 0 slopes."""
    pred = apply(model, scores).set_index("id").join(slopes.set_index("id")[["set", "slope", "p_positive"]], how="inner").dropna(subset=["slope"])
    rng = np.random.default_rng(params.seed)
    return correlations(pred.predicted_slope.to_numpy(), pred.slope.to_numpy(), pred["set"].to_numpy(), labels(pred, params).to_numpy(), rng)
