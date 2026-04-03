"""Evaluation metrics for MIA models."""

from typing import Optional, Tuple

import numpy as np


def compute_curve_metrics(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute AUC, TPR@1%FPR, and TPR@0.1%FPR.

    Args:
        labels_np: Ground truth labels.
        probs_np: Predicted probabilities.

    Returns:
        Tuple of (auc, tpr_at_1pct_fpr, tpr_at_01pct_fpr) or (None, None, None) on error.
    """
    try:
        probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=1.0, neginf=0.0)
        from sklearn.metrics import roc_auc_score, roc_curve

        auc = roc_auc_score(labels_np, probs_np)
        fpr, tpr, _ = roc_curve(labels_np, probs_np)

        def tpr_at(fpr_target: float) -> float:
            idx = np.searchsorted(fpr, fpr_target, side="right")
            if idx == 0:
                return float(tpr[0])
            if idx >= len(fpr):
                return float(tpr[-1])
            f0, f1 = fpr[idx - 1], fpr[idx]
            t0, t1 = tpr[idx - 1], tpr[idx]
            if f1 == f0:
                return float(max(t0, t1))
            return float(t0 + (fpr_target - f0) * (t1 - t0) / (f1 - f0))

        return auc, tpr_at(0.01), tpr_at(0.001)
    except Exception:
        return None, None, None
