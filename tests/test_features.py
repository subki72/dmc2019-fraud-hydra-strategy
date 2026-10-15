"""Unit tests for the Scikit-Learn compatible FraudFeatureEngineer transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from dmc_fraud.data.schema import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS
from dmc_fraud.features.pipeline import FraudFeatureEngineer


def test_feature_engineer_transform_dataframe(synthetic_train_df: pd.DataFrame) -> None:
    """Verify that transforming a DataFrame produces all 9 engineered features."""
    fe = FraudFeatureEngineer()
    fe.fit(synthetic_train_df)

    transformed_df = fe.transform(synthetic_train_df)

    assert isinstance(transformed_df, pd.DataFrame)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in transformed_df.columns, f"Engineered column {col} is missing"

    assert len(transformed_df) == len(synthetic_train_df)


def test_feature_engineer_mathematical_correctness() -> None:
    """Verify mathematical formulas for reconstructed items, ratios, and interactions."""
    raw_df = pd.DataFrame(
        [
            {
                "trustLevel": 2,
                "totalScanTimeInSeconds": 100.0,
                "grandTotal": 50.0,
                "lineItemVoids": 2,
                "scansWithoutRegistration": 1,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": 0.1,  # 0.1 * 100 = 10 items
                "valuePerSecond": 0.5,
                "lineItemVoidsPerPosition": 0.2,
            }
        ]
    )

    fe = FraudFeatureEngineer(epsilon=1e-6)
    res = fe.fit_transform(raw_df)

    # 1. scannedLineItemsTotal = round(0.1 * 100) = 10
    assert res.loc[0, "scannedLineItemsTotal"] == 10

    # 2. avgTimePerItem = 100 / (10 + 1e-6) ≈ 9.999999
    assert pytest.approx(res.loc[0, "avgTimePerItem"], rel=1e-4) == 10.0

    # 3. voidsPerTotalItem = 2 / (10 + 1e-6) ≈ 0.2
    assert pytest.approx(res.loc[0, "voidsPerTotalItem"], rel=1e-4) == 0.2

    # 4. pricePerItem = 50 / 10 ≈ 5.0
    assert pytest.approx(res.loc[0, "pricePerItem"], rel=1e-4) == 5.0

    # 5. log_grandTotal = log1p(50)
    assert pytest.approx(res.loc[0, "log_grandTotal"]) == np.log1p(50.0)

    # 6. interaction_trust_time = 2 * log1p(100)
    assert pytest.approx(res.loc[0, "interaction_trust_time"]) == 2 * np.log1p(100.0)

    # 7. interaction_void_price = 2 * 5.0 = 10.0
    assert pytest.approx(res.loc[0, "interaction_void_price"], rel=1e-4) == 10.0


def test_feature_engineer_zero_item_division_stability() -> None:
    """Verify numerical stability when total items scanned is 0 (epsilon division protection)."""
    zero_df = pd.DataFrame(
        [
            {
                "trustLevel": 1,
                "totalScanTimeInSeconds": 0.0,
                "grandTotal": 0.0,
                "lineItemVoids": 0,
                "scansWithoutRegistration": 0,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": 0.0,
                "valuePerSecond": 0.0,
                "lineItemVoidsPerPosition": 0.0,
            }
        ]
    )

    fe = FraudFeatureEngineer(epsilon=1e-6)
    res = fe.fit_transform(zero_df)

    # Should not raise ZeroDivisionError and should not produce inf or NaN
    assert not np.isnan(res.loc[0, "avgTimePerItem"])
    assert not np.isinf(res.loc[0, "avgTimePerItem"])
    assert res.loc[0, "scannedLineItemsTotal"] == 0


def test_feature_engineer_inf_nan_replacement() -> None:
    """Verify that any accidental Inf or NaN values are safely coerced to 0.0."""
    edge_df = pd.DataFrame(
        [
            {
                "trustLevel": 1,
                "totalScanTimeInSeconds": 10.0,
                "grandTotal": 10.0,
                "lineItemVoids": 0,
                "scansWithoutRegistration": 0,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": 1.0,
                "valuePerSecond": np.nan,  # NaN in raw input
                "lineItemVoidsPerPosition": 0.0,
            }
        ]
    )

    fe = FraudFeatureEngineer()
    res = fe.fit_transform(edge_df)
    assert not res.isna().any().any()


def test_feature_engineer_sklearn_clone() -> None:
    """Verify transformer is fully compatible with Scikit-Learn clone."""
    fe = FraudFeatureEngineer(epsilon=1e-5)
    cloned_fe = clone(fe)
    assert cloned_fe.epsilon == 1e-5
    assert isinstance(cloned_fe, FraudFeatureEngineer)


def test_feature_engineer_numpy_array_input(synthetic_train_df: pd.DataFrame) -> None:
    """Verify transformer operates on raw NumPy matrices."""
    X_raw = synthetic_train_df[RAW_FEATURE_COLUMNS].values
    fe = FraudFeatureEngineer()
    fe.fit(X_raw)
    out_df = fe.transform(X_raw)
    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) == len(X_raw)


def test_feature_engineer_missing_column_error(synthetic_train_df: pd.DataFrame) -> None:
    """Transforming a DataFrame with missing columns should raise ValueError."""
    bad_df = synthetic_train_df.drop(columns=["trustLevel"])
    fe = FraudFeatureEngineer()
    with pytest.raises(ValueError, match="Missing required raw columns"):
        fe.transform(bad_df)


def test_feature_engineer_get_feature_names_out(synthetic_train_df: pd.DataFrame) -> None:
    """Verify get_feature_names_out conforms to scikit-learn ndarray return contract."""
    fe = FraudFeatureEngineer()
    fe.fit(synthetic_train_df)
    names = fe.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert len(names) == 18


def test_feature_engineer_zero_items_positive_time_no_extreme_outlier() -> None:
    """Verify that zero scanned items with positive scan time and grand total yields 0.0 instead of extreme outliers."""
    edge_df = pd.DataFrame(
        [
            {
                "trustLevel": 1,
                "totalScanTimeInSeconds": 500.0,
                "grandTotal": 250.0,
                "lineItemVoids": 3,
                "scansWithoutRegistration": 2,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": 0.0,  # 0.0 * 500 = 0 items
                "valuePerSecond": 0.5,
                "lineItemVoidsPerPosition": 0.0,
            }
        ]
    )

    fe = FraudFeatureEngineer()
    res = fe.fit_transform(edge_df)

    assert res.loc[0, "scannedLineItemsTotal"] == 0
    assert res.loc[0, "avgTimePerItem"] == 0.0
    assert res.loc[0, "pricePerItem"] == 0.0
    assert res.loc[0, "voidsPerTotalItem"] == 0.0
    assert res.loc[0, "noRegPerItem"] == 0.0


def test_feature_config_parameterization() -> None:
    """Verify FeatureConfig encapsulates epsilon and integrates with FeatureEngineer."""
    from dmc_fraud.config import FeatureConfig

    cfg = FeatureConfig(epsilon=1e-4)
    assert cfg.epsilon == 1e-4

    fe = FraudFeatureEngineer(epsilon=cfg.epsilon)
    assert fe.epsilon == 1e-4


def test_feature_engineer_nan_in_scanned_items_per_second() -> None:
    """Verify that NaN or Inf in rate or time columns does not crash with IntCastingNaNError."""
    nan_df = pd.DataFrame(
        [
            {
                "trustLevel": 1,
                "totalScanTimeInSeconds": 10.0,
                "grandTotal": 10.0,
                "lineItemVoids": 0,
                "scansWithoutRegistration": 0,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": np.nan,
                "valuePerSecond": np.inf,
                "lineItemVoidsPerPosition": 0.0,
            }
        ]
    )

    fe = FraudFeatureEngineer()
    res = fe.fit_transform(nan_df)
    assert res.loc[0, "scannedLineItemsTotal"] == 0
    assert res.loc[0, "avgTimePerItem"] == 0.0
    assert not res.isna().any().any()


def test_feature_config_optional_none_epsilon() -> None:
    """Verify FeatureConfig accepts epsilon=None and FraudFeatureEngineer processes cleanly."""
    from dmc_fraud.config import FeatureConfig

    cfg = FeatureConfig(epsilon=None)
    assert cfg.epsilon is None

    fe = FraudFeatureEngineer(epsilon=cfg.epsilon)
    assert fe.epsilon is None

    df = pd.DataFrame(
        [
            {
                "trustLevel": 1,
                "totalScanTimeInSeconds": 100.0,
                "grandTotal": 50.0,
                "lineItemVoids": 2,
                "scansWithoutRegistration": 1,
                "quantityModifications": 0,
                "scannedLineItemsPerSecond": 0.1,  # 10 items
                "valuePerSecond": 0.5,
                "lineItemVoidsPerPosition": 0.2,
            }
        ]
    )
    res = fe.fit_transform(df)
    assert res.loc[0, "scannedLineItemsTotal"] == 10
    # With epsilon=None, denom is 10.0 exactly
    assert res.loc[0, "avgTimePerItem"] == 10.0
    assert res.loc[0, "pricePerItem"] == 5.0
