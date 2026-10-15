"""Unit tests for the HydraClassifier ensemble architecture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmc_fraud.data.schema import TARGET_COLUMN
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.hydra import HydraClassifier


def test_hydra_classifier_fit_predict(synthetic_train_df: pd.DataFrame) -> None:
    """Verify that HydraClassifier fits and generates predictions and probabilities."""
    fe = FraudFeatureEngineer()
    X_train_adv = fe.fit_transform(synthetic_train_df)
    y_train = synthetic_train_df[TARGET_COLUMN].values

    # Use fast hyperparameters for test execution speed
    model = HydraClassifier(
        xgb_params={"n_estimators": 10, "max_depth": 3, "eval_metric": "logloss", "random_state": 42},
        rf_params={"n_estimators": 10, "max_depth": 3, "random_state": 42, "n_jobs": 1},
        thresholds={1: 0.80, 2: 0.70},
    )

    model.fit(X_train_adv, y_train)
    assert model.is_fitted_
    assert model.voting_model_ is not None

    # Test predict_proba
    probs = model.predict_proba(X_train_adv)
    assert probs.shape == (len(synthetic_train_df), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test predict
    preds = model.predict(X_train_adv)
    assert len(preds) == len(synthetic_train_df)
    assert set(preds).issubset({0, 1})


def test_hydra_classifier_dual_gate_segmentation() -> None:
    """Verify dual-gate decision boundary: trustLevel 1 uses t1, level 2 uses t2, others 0."""
    df_test = pd.DataFrame(
        {
            "trustLevel": [1, 1, 2, 2, 3, 4],
            "feature_1": [1.0] * 6,
        }
    )

    model = HydraClassifier(thresholds={1: 0.85, 2: 0.75})
    model.is_fitted_ = True

    # Mock predict_proba to return deterministic probabilities: [0.80, 0.90, 0.70, 0.80, 0.95, 0.99]
    mock_probs = np.array(
        [
            [0.20, 0.80],  # L1, prob 0.80 (< 0.85) => 0
            [0.10, 0.90],  # L1, prob 0.90 (>= 0.85) => 1
            [0.30, 0.70],  # L2, prob 0.70 (< 0.75) => 0
            [0.20, 0.80],  # L2, prob 0.80 (>= 0.75) => 1
            [0.05, 0.95],  # L3, prob 0.95 (level > 2) => 0
            [0.01, 0.99],  # L4, prob 0.99 (level > 2) => 0
        ]
    )

    model.predict_proba = lambda X: mock_probs  # type: ignore[assignment]

    preds = model.predict(df_test)
    expected = np.array([0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(preds, expected)


def test_hydra_classifier_unfitted_raises() -> None:
    """Calling predict or predict_proba on unfitted model raises RuntimeError."""
    model = HydraClassifier()
    dummy_data = pd.DataFrame({"trustLevel": [1], "col": [1.0]})

    with pytest.raises(RuntimeError, match="not fitted yet"):
        model.predict_proba(dummy_data)


def test_hydra_classifier_filter_danger_levels(synthetic_train_df: pd.DataFrame) -> None:
    """Verify filter_danger_on_train restricts training to specified trust levels."""
    fe = FraudFeatureEngineer()
    X_train_adv = fe.fit_transform(synthetic_train_df)
    y_train = synthetic_train_df[TARGET_COLUMN].values

    model = HydraClassifier(
        xgb_params={"n_estimators": 5, "max_depth": 2, "eval_metric": "logloss", "random_state": 42},
        rf_params={"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1},
        danger_trust_levels=[1],
        filter_danger_on_train=True,
    )
    model.fit(X_train_adv, y_train)
    assert model.is_fitted_


def test_hydra_classifier_single_class_raises(synthetic_train_df: pd.DataFrame) -> None:
    """Fitting HydraClassifier with only 1 class must raise ValueError."""
    fe = FraudFeatureEngineer()
    X_train_adv = fe.fit_transform(synthetic_train_df)
    y_zeros = np.zeros(len(synthetic_train_df), dtype=int)

    model = HydraClassifier()
    with pytest.raises(ValueError, match="must contain both classes"):
        model.fit(X_train_adv, y_zeros)


def test_hydra_classifier_missing_features_predict_raises(synthetic_train_df: pd.DataFrame) -> None:
    """Predicting on DataFrame missing expected features must raise ValueError."""
    fe = FraudFeatureEngineer()
    X_train_adv = fe.fit_transform(synthetic_train_df)
    y_train = synthetic_train_df[TARGET_COLUMN].values

    model = HydraClassifier(
        xgb_params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
        rf_params={"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1},
    )
    model.fit(X_train_adv, y_train)

    bad_df = X_train_adv.drop(columns=["pricePerItem"])
    with pytest.raises(ValueError, match="missing required features"):
        model.predict_proba(bad_df)


def test_hydra_classifier_predict_with_probs() -> None:
    """Verify predict_with_probs applies thresholds directly to probability array."""
    model = HydraClassifier(thresholds={1: 0.85, 2: 0.75})
    df = pd.DataFrame({"trustLevel": [1, 1, 2, 2, 3]})
    probs = np.array([0.90, 0.80, 0.80, 0.70, 0.95])
    preds = model.predict_with_probs(probs, df)
    np.testing.assert_array_equal(preds, np.array([1, 0, 1, 0, 0]))


def test_hydra_classifier_check_is_fitted_validation(synthetic_train_df: pd.DataFrame) -> None:
    """Verify sklearn.utils.validation.check_is_fitted recognizes model state on ensemble_."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    from dmc_fraud.models.hydra import HydraNotFittedError

    model = HydraClassifier()
    # Unfitted check raises NotFittedError and HydraNotFittedError
    with pytest.raises(NotFittedError):
        check_is_fitted(model, "ensemble_")

    with pytest.raises(HydraNotFittedError):
        model.predict_proba(synthetic_train_df)

    # After fit, check_is_fitted succeeds without raising
    fe = FraudFeatureEngineer()
    X = fe.fit_transform(synthetic_train_df)
    y = synthetic_train_df[TARGET_COLUMN].values
    model = HydraClassifier(
        xgb_params={"n_estimators": 5, "max_depth": 2, "random_state": 42},
        rf_params={"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1},
    )
    model.fit(X, y)
    check_is_fitted(model, "ensemble_")
