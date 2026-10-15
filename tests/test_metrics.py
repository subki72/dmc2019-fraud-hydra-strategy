"""Unit tests for DMC 2019 profit matrix and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dmc_fraud.evaluation.metrics import calculate_dmc_profit, evaluate_predictions


def test_calculate_dmc_profit_individual_cases() -> None:
    """Verify DMC 2019 profit equation for pure TP, FP, FN, and TN cases.

    Matrix rules:
    - TP: +5 €
    - FP: -25 €
    - FN: -5 €
    - TN: 0 €
    """
    # 1. Pure True Positive: caught 1 fraud
    assert calculate_dmc_profit([1], [1]) == 5.0

    # 2. Pure False Positive: falsely accused innocent customer
    assert calculate_dmc_profit([0], [1]) == -25.0

    # 3. Pure False Negative: missed fraud
    assert calculate_dmc_profit([1], [0]) == -5.0

    # 4. Pure True Negative: clean customer passed
    assert calculate_dmc_profit([0], [0]) == 0.0


def test_calculate_dmc_profit_composite() -> None:
    """Verify composite confusion matrix profit calculation."""
    # 10 TP (+50), 2 FP (-50), 4 FN (-20), 100 TN (+0) => Total = -20
    y_true = [1] * 10 + [0] * 2 + [1] * 4 + [0] * 100
    y_pred = [1] * 10 + [1] * 2 + [0] * 4 + [0] * 100

    profit = calculate_dmc_profit(y_true, y_pred)
    assert profit == -20.0


def test_calculate_dmc_profit_custom_costs() -> None:
    """Verify profit calculation with customized cost parameters."""
    # TP=+10, FP=-50, FN=-10, TN=+1
    profit = calculate_dmc_profit(
        y_true=[1, 0, 1, 0],
        y_pred=[1, 1, 0, 0],
        tp_reward=10.0,
        fp_penalty=50.0,
        fn_penalty=10.0,
        tn_reward=1.0,
    )
    # TP=1 (+10), FP=1 (-50), FN=1 (-10), TN=1 (+1) => -49
    assert profit == -49.0


def test_calculate_dmc_profit_empty_input_raises() -> None:
    """Empty inputs should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot calculate profit on empty arrays"):
        calculate_dmc_profit([], [])


def test_calculate_dmc_profit_length_mismatch_raises() -> None:
    """Mismatched array lengths should raise ValueError."""
    with pytest.raises(ValueError, match="Array length mismatch"):
        calculate_dmc_profit([1, 0], [1])


def test_evaluate_predictions_comprehensive() -> None:
    """Test full evaluation report structure and metric values."""
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1])
    # TP=1, FN=1, TN=2, FP=1
    # Profit: (1 * 5) - (1 * 25) - (1 * 5) + 0 = -25

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["profit"] == -25.0
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["total_samples"] == 5
    assert metrics["total_frauds"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.6
    assert metrics["max_possible_profit"] == 10.0  # 2 * 5
    assert metrics["profit_efficiency"] == -2.5


def test_evaluate_predictions_empty_and_mismatch() -> None:
    """Test error handling in evaluate_predictions."""
    with pytest.raises(ValueError, match="Cannot evaluate predictions on empty arrays"):
        evaluate_predictions([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions([1, 0], [1])


def test_calculate_dmc_profit_non_binary_labels_raises() -> None:
    """Non-binary labels in y_true or y_pred must raise ValueError."""
    with pytest.raises(ValueError, match="y_true contains invalid non-binary values"):
        calculate_dmc_profit([0, 1, 2], [0, 1, 0])

    with pytest.raises(ValueError, match="y_pred contains invalid non-binary values"):
        calculate_dmc_profit([0, 1, 0], [0, -1, 1])


def test_evaluate_predictions_with_reward_tn() -> None:
    """Custom cost matrix with positive TN reward must be factored into max_possible_profit."""
    from dmc_fraud.config import CostMatrixConfig

    costs = CostMatrixConfig(profit_tp=5.0, reward_tn=1.0)
    metrics = evaluate_predictions([1, 0, 0], [1, 0, 0], costs=costs)
    # 1 fraud (+5) + 2 clean (+2) = 7.0
    assert metrics["max_possible_profit"] == 7.0
    assert metrics["profit"] == 7.0
    assert metrics["profit_efficiency"] == 1.0
