"""Compare two scorers on the same questions: feature by feature, and in the predictions the fitted regression makes from their scores.

The reference scorer (`a`, normally gemini-flash) is the one the regression was trained on. A question counts only when both
scorers answered it. Feature agreement is measured on the scores (0 or 1 with one sample per question), with Cohen's kappa
for the agreement beyond chance. Prediction agreement is the correlation of the predicted slopes, the share of questions in
the same class, and the gap in percentile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from qscorer.predict import apply


def kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's kappa of two binary vectors (scores above 0.5 count as present); None when both raters are constant."""
    a, b = a > 0.5, b > 0.5
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe >= 1 else (po - pe) / (1 - pe)


def compare(model: dict, scores_a: pd.DataFrame, scores_b: pd.DataFrame, a: str = "a", b: str = "b") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per question, per feature, and summary comparison of two `run_presence` score tables."""
    item_ids = [it["id"] for it in model["items"]]
    names = {it["id"]: it["name"] for it in model["items"]}
    coef = model["coefficients"]
    ids = [q for q in scores_a.id if q in set(scores_b.id)]
    sa = scores_a.set_index("id").loc[ids, item_ids]
    sb = scores_b.set_index("id").loc[ids, item_ids]
    pa = apply(model, sa.reset_index()).set_index("id")
    pb = apply(model, sb.reset_index()).set_index("id")

    gap = (sa - sb).abs()
    questions = pd.DataFrame({
        "id": ids,
        f"slope_{a}": pa.predicted_slope.to_numpy(), f"slope_{b}": pb.predicted_slope.to_numpy(),
        f"percentile_{a}": pa.percentile.to_numpy(), f"percentile_{b}": pb.percentile.to_numpy(),
        f"class_{a}": pa["class"].to_numpy(), f"class_{b}": pb["class"].to_numpy(),
        "same_class": (pa["class"] == pb["class"]).to_numpy(),
        "slope_gap": (pb.predicted_slope - pa.predicted_slope).to_numpy(),
        "features_disagreeing": (gap > 0.5).sum(axis=1).to_numpy(),
        "disagreeing": [", ".join(f for f in item_ids if gap.loc[q, f] > 0.5) for q in ids],
    })

    rows = []
    for f in item_ids:
        x, y = sa[f].to_numpy(float), sb[f].to_numpy(float)
        k = kappa(x, y) if len(ids) else None
        rows.append({"feature": f, "name": names[f], "coefficient": coef[f],
                     f"prevalence_{a}": float(x.mean()) if len(ids) else None, f"prevalence_{b}": float(y.mean()) if len(ids) else None,
                     "agreement": float(((x > 0.5) == (y > 0.5)).mean()) if len(ids) else None, "kappa": k,
                     # the slope this feature's disagreement moves, averaged over the questions: what it costs the prediction
                     "mean_slope_effect": float(np.mean(np.abs(x - y)) * abs(coef[f])) if len(ids) else None})
    features = pd.DataFrame(rows).sort_values("mean_slope_effect", ascending=False)

    def corr(fn, u, v):
        return float(fn(u, v)[0]) if len(u) >= 3 and np.std(u) > 0 and np.std(v) > 0 else None

    ks = [k for k in features.kappa if k is not None and not pd.isna(k)]
    summary = {
        "reference": a, "candidate": b,
        "questions_compared": len(ids),
        "only_in_reference": sorted(set(scores_a.id) - set(ids)), "only_in_candidate": sorted(set(scores_b.id) - set(ids)),
        "predictions": {
            "spearman": corr(spearmanr, pa.predicted_slope.to_numpy(), pb.predicted_slope.to_numpy()),
            "pearson": corr(pearsonr, pa.predicted_slope.to_numpy(), pb.predicted_slope.to_numpy()),
            "same_class": float(questions.same_class.mean()) if len(ids) else None,
            "mean_abs_slope_gap": float(questions.slope_gap.abs().mean()) if len(ids) else None,
            "mean_slope_gap": float(questions.slope_gap.mean()) if len(ids) else None,
            "mean_abs_percentile_gap": float((pa.percentile - pb.percentile).abs().mean()) if len(ids) else None,
            "training_slope_sd": float(np.std(model["training_predictions"])),
        },
        "features": {
            "mean_agreement": float(features.agreement.mean()) if len(ids) else None,
            "mean_kappa": float(np.mean(ks)) if ks else None,
            "features_with_kappa_below_0.4": int(sum(k < 0.4 for k in ks)),
            "mean_features_disagreeing_per_question": float(questions.features_disagreeing.mean()) if len(ids) else None,
        },
    }
    return questions, features, summary
