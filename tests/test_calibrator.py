"""Unit tests for Stratified K-Fold Out-of-Fold Dual-Gate Calibrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmc_fraud.config import DMCConfig
from dmc_fraud.data.schema import TARGET_COLUMN
from dmc_fraud.evaluation.calibrator import DualGateCalibrator


def test_calibrator_out_of_fold_execution(synthetic_train_df: pd.DataFrame) -> None:
    """Verify that Out-of-Fold calibration executes cleanly and selects optimal thresholds."""
    cfg = DMCConfig()
    # Fast test hyperparameters
    cfg.models.xgb_params = {"n_estimators": 10, "max_depth": 3, "eval_metric": "logloss", "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1}

    calibrator = DualGateCalibrator(
        config=cfg,
        cv_folds=3,
        threshold_min=0.60,
        threshold_max=0.95,
        threshold_step=0.10,
        random_state=42,
    )

    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    y_train = synthetic_train_df[TARGET_COLUMN].values

    res = calibrator.calibrate(X_train, y_train)

    # 1. Output result checks
    assert 0.60 <= res.best_t1 <= 0.95
    assert 0.60 <= res.best_t2 <= 0.95
    assert isinstance(res.best_profit, float)

    # 2. Verify OOF probabilities dimensions match full training dataset (No Leakage check)
    assert res.oof_probabilities is not None
    assert len(res.oof_probabilities) == len(X_train)

    # 3. Verify threshold search history was recorded
    assert len(res.threshold_history) > 0
    assert any(h["profit"] == res.best_profit for h in res.threshold_history)

    # 4. Metrics verification
    assert "true_positives" in res.oof_metrics
    assert "false_positives" in res.oof_metrics
    assert res.oof_metrics["total_samples"] == len(X_train)


def test_calibrator_missing_trust_level_raises(synthetic_train_df: pd.DataFrame) -> None:
    """Input DataFrame without trustLevel must raise ValueError."""
    bad_df = synthetic_train_df.drop(columns=["trustLevel", TARGET_COLUMN])
    y_train = synthetic_train_df[TARGET_COLUMN].values

    calibrator = DualGateCalibrator()
    with pytest.raises(ValueError, match="Input DataFrame must include 'trustLevel'"):
        calibrator.calibrate(bad_df, y_train)


def test_calibrator_mismatched_length_raises(synthetic_train_df: pd.DataFrame) -> None:
    """Mismatched feature and target lengths must raise ValueError."""
    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    calibrator = DualGateCalibrator()

    with pytest.raises(ValueError, match="does not match y_train length"):
        calibrator.calibrate(X_train, [0, 1])


def test_calibrator_single_class_raises(synthetic_train_df: pd.DataFrame) -> None:
    """Calibrating on single-class data must raise ValueError."""
    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    y_all_clean = np.zeros(len(synthetic_train_df), dtype=int)

    calibrator = DualGateCalibrator()
    with pytest.raises(ValueError, match="dataset must contain both classes"):
        calibrator.calibrate(X_train, y_all_clean)


def test_calibrator_dynamic_fold_adjustment(synthetic_train_df: pd.DataFrame) -> None:
    """When minority class count is less than requested cv_folds, calibrator dynamically adjusts folds."""
    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN]).copy()
    # Create dataset with exactly 2 frauds
    y_few = np.zeros(len(synthetic_train_df), dtype=int)
    y_few[0] = 1
    y_few[1] = 1

    cfg = DMCConfig()
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}

    calibrator = DualGateCalibrator(config=cfg, cv_folds=5)
    res = calibrator.calibrate(X_train, y_few)
    assert res.best_profit is not None
    assert res.oof_probabilities is not None
    assert len(res.oof_probabilities) == len(X_train)


def test_calibrator_independent_search_and_conservative_tie_breaking(synthetic_train_df: pd.DataFrame) -> None:
    """Verify conservative tie-breaking favors higher threshold on equal profit."""
    from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS

    cfg = DMCConfig()
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}

    calibrator = DualGateCalibrator(
        config=cfg,
        cv_folds=2,
        threshold_min=0.50,
        threshold_max=0.90,
        threshold_step=0.10,
    )

    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    y_train = synthetic_train_df[TARGET_COLUMN].values

    res = calibrator.calibrate(X_train, y_train)
    assert 0.50 <= res.best_t1 <= 0.90
    assert 0.50 <= res.best_t2 <= 0.90
    assert len(res.threshold_history) > 0

    # Test missing trust level 2 fallback to default
    df_only_l1 = synthetic_train_df.copy()
    df_only_l1.loc[df_only_l1["trustLevel"] == 2, "trustLevel"] = 3
    X_l1 = df_only_l1.drop(columns=[TARGET_COLUMN])
    y_l1 = df_only_l1[TARGET_COLUMN].values

    res_l1 = calibrator.calibrate(X_l1, y_l1)
    assert res_l1.best_t2 == DEFAULT_CALIBRATED_THRESHOLDS[2]


def test_calibrator_threshold_range_respects_threshold_max(synthetic_train_df: pd.DataFrame) -> None:
    """Verify threshold evaluation strictly respects threshold_max and validates bounds."""
    cfg = DMCConfig()
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}

    # threshold_max=0.99 with step 0.05 must not evaluate 1.00
    calibrator = DualGateCalibrator(
        config=cfg,
        cv_folds=2,
        threshold_min=0.50,
        threshold_max=0.99,
        threshold_step=0.05,
    )
    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    y_train = synthetic_train_df[TARGET_COLUMN].values

    res = calibrator.calibrate(X_train, y_train)
    assert res.best_t1 <= 0.99
    assert res.best_t2 <= 0.99
    assert all(h["t1"] <= 0.99 and h["t2"] <= 0.99 for h in res.threshold_history)

    # Invalid bounds must raise ValueError
    with pytest.raises(ValueError, match="cannot exceed threshold_max"):
        DualGateCalibrator(threshold_min=0.90, threshold_max=0.50)

    with pytest.raises(ValueError, match="must be positive"):
        DualGateCalibrator(threshold_step=0.0)


def test_calibrator_wires_config_epsilon(synthetic_train_df: pd.DataFrame) -> None:
    """Verify calibrator passes custom feature epsilon into fold feature transformers."""
    cfg = DMCConfig()
    cfg.features.epsilon = 1e-4
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}

    calibrator = DualGateCalibrator(config=cfg, cv_folds=2)
    X_train = synthetic_train_df.drop(columns=[TARGET_COLUMN])
    y_train = synthetic_train_df[TARGET_COLUMN].values

    res = calibrator.calibrate(X_train, y_train)
    assert res.best_profit is not None
