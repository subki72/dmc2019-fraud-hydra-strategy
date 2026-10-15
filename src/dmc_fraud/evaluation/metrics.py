"""Evaluation metrics for the DMC 2019 Retail Fraud Detection cost matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dmc_fraud.config import CostMatrixConfig


def calculate_dmc_profit(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    tp_reward: float = 5.0,
    fp_penalty: float = 25.0,
    fn_penalty: float = 5.0,
    tn_reward: float = 0.0,
) -> float:
    """Calculate total economic profit under the Data Mining Cup 2019 asymmetric cost matrix.

    DMC 2019 Profit Equation:
        Profit = (TP * 5€) - (FP * 25€) - (FN * 5€) + (TN * 0€)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary fraud labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary fraud labels (0 or 1).
    tp_reward : float, default=5.0
        Euro gain per caught fraud.
    fp_penalty : float, default=25.0
        Euro loss per false accusation (customer friction & manual audit cost).
    fn_penalty : float, default=5.0
        Euro loss per uncaught fraud.
    tn_reward : float, default=0.0
        Euro gain per correctly cleared transaction.

    Returns
    -------
    float
        Total monetary profit in Euros.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    if len(y_t) == 0:
        raise ValueError("Cannot calculate profit on empty arrays.")

    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)} items, y_pred has {len(y_p)} items.")

    invalid_t = np.setdiff1d(y_t, [0, 1])
    if len(invalid_t) > 0:
        raise ValueError(f"y_true contains invalid non-binary values: {invalid_t.tolist()}. Expected only 0 or 1.")

    invalid_p = np.setdiff1d(y_p, [0, 1])
    if len(invalid_p) > 0:
        raise ValueError(f"y_pred contains invalid non-binary values: {invalid_p.tolist()}. Expected only 0 or 1.")

    # Compute confusion matrix with explicit labels [0, 1] to handle single-class predictions
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    profit = (tp * tp_reward) - (fp * fp_penalty) - (fn * fn_penalty) + (tn * tn_reward)
    return float(profit)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    costs: CostMatrixConfig | None = None,
) -> dict[str, Any]:
    """Compute comprehensive performance and economic metrics for fraud predictions.

    Parameters
    ----------
    y_true : array-like
        True binary fraud labels.
    y_pred : array-like
        Predicted binary fraud labels.
    costs : Optional[CostMatrixConfig]
        Cost matrix configuration. Defaults to DMC 2019 standard costs.

    Returns
    -------
    dict
        Dictionary containing profit, confusion matrix components, and standard ML metrics.
    """
    cost_cfg = costs or CostMatrixConfig()

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    if len(y_t) == 0:
        raise ValueError("Cannot evaluate predictions on empty arrays.")

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: {len(y_t)} vs {len(y_p)}")

    profit = calculate_dmc_profit(
        y_t,
        y_p,
        tp_reward=cost_cfg.profit_tp,
        fp_penalty=cost_cfg.penalty_fp,
        fn_penalty=cost_cfg.penalty_fn,
        tn_reward=cost_cfg.reward_tn,
    )

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_samples = len(y_t)
    total_frauds = int(np.sum(y_t == 1))
    total_clean = int(np.sum(y_t == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0

    max_possible_profit = float(total_frauds * cost_cfg.profit_tp + total_clean * cost_cfg.reward_tn)
    profit_efficiency = float(profit / max_possible_profit) if max_possible_profit > 0 else 0.0

    return {
        "profit": profit,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_samples": total_samples,
        "total_frauds": total_frauds,
        "total_clean": total_clean,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "max_possible_profit": max_possible_profit,
        "profit_efficiency": profit_efficiency,
    }
