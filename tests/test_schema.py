"""Unit tests for DMC 2019 schema validation and record schemas."""

from __future__ import annotations

import pandas as pd
import pytest

from dmc_fraud.data.schema import (
    SchemaValidationError,
    TransactionRecord,
    validate_dataframe,
)


def test_validate_dataframe_valid(synthetic_train_df: pd.DataFrame) -> None:
    """Valid DataFrame should pass validation without raising."""
    clean_df = validate_dataframe(synthetic_train_df, require_target=True)
    assert not clean_df.empty
    assert len(clean_df) == len(synthetic_train_df)
    assert clean_df["trustLevel"].dtype == int


def test_validate_dataframe_missing_feature_column(synthetic_train_df: pd.DataFrame) -> None:
    """DataFrame missing a required feature column must raise SchemaValidationError."""
    invalid_df = synthetic_train_df.drop(columns=["grandTotal"])
    with pytest.raises(SchemaValidationError, match="Missing required raw feature columns"):
        validate_dataframe(invalid_df)


def test_validate_dataframe_missing_target_column(synthetic_test_df: pd.DataFrame) -> None:
    """DataFrame missing target column when require_target=True must raise SchemaValidationError."""
    with pytest.raises(SchemaValidationError, match="Missing required target column: 'fraud'"):
        validate_dataframe(synthetic_test_df, require_target=True)


def test_validate_dataframe_invalid_trust_level(synthetic_train_df: pd.DataFrame) -> None:
    """trustLevel values outside [1..6] must trigger a validation error."""
    invalid_df = synthetic_train_df.copy()
    invalid_df.loc[0, "trustLevel"] = 99  # Invalid trust level
    with pytest.raises(SchemaValidationError, match="'trustLevel' must be in range"):
        validate_dataframe(invalid_df)


def test_validate_dataframe_negative_values(synthetic_train_df: pd.DataFrame) -> None:
    """Negative values in strictly non-negative columns must trigger a validation error."""
    invalid_df = synthetic_train_df.copy()
    invalid_df.loc[0, "grandTotal"] = -10.5
    with pytest.raises(SchemaValidationError, match="contains .* negative value"):
        validate_dataframe(invalid_df)


def test_validate_dataframe_null_values(synthetic_train_df: pd.DataFrame) -> None:
    """Null values in features must trigger validation error."""
    invalid_df = synthetic_train_df.copy()
    invalid_df.loc[0, "totalScanTimeInSeconds"] = None
    with pytest.raises(SchemaValidationError, match="Null values detected"):
        validate_dataframe(invalid_df)


def test_validate_dataframe_empty_or_non_dataframe() -> None:
    """Empty inputs or non-DataFrames must raise SchemaValidationError."""
    with pytest.raises(SchemaValidationError, match="Input must be a valid pandas DataFrame"):
        validate_dataframe("not a dataframe")  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationError, match="Input DataFrame is empty"):
        validate_dataframe(pd.DataFrame())


def test_transaction_record_pydantic_valid() -> None:
    """Pydantic model validation on a valid single transaction."""
    record = TransactionRecord(
        trustLevel=1,
        totalScanTimeInSeconds=120.0,
        grandTotal=45.5,
        lineItemVoids=1,
        scansWithoutRegistration=0,
        quantityModifications=1,
        scannedLineItemsPerSecond=0.1,
        valuePerSecond=0.38,
        lineItemVoidsPerPosition=0.08,
        fraud=0,
    )
    assert record.trustLevel == 1
    assert record.fraud == 0


def test_transaction_record_pydantic_invalid() -> None:
    """Pydantic model validation rejects out-of-range trustLevel."""
    with pytest.raises(ValueError):
        TransactionRecord(
            trustLevel=7,  # Max allowed is 6
            totalScanTimeInSeconds=120.0,
            grandTotal=45.5,
            lineItemVoids=1,
            scansWithoutRegistration=0,
            quantityModifications=1,
            scannedLineItemsPerSecond=0.1,
            valuePerSecond=0.38,
            lineItemVoidsPerPosition=0.08,
        )


def test_validate_dataframe_non_integer_discrete_columns(synthetic_train_df: pd.DataFrame) -> None:
    """Non-integer float values in discrete integer columns must raise SchemaValidationError."""
    invalid_df = synthetic_train_df.copy()
    invalid_df["trustLevel"] = invalid_df["trustLevel"].astype(object)
    invalid_df.loc[0, "trustLevel"] = 1.8
    with pytest.raises(SchemaValidationError, match="must contain integer values"):
        validate_dataframe(invalid_df)

    invalid_df2 = synthetic_train_df.copy()
    invalid_df2["lineItemVoids"] = invalid_df2["lineItemVoids"].astype(object)
    invalid_df2.loc[0, "lineItemVoids"] = 2.5
    with pytest.raises(SchemaValidationError, match="must contain integer values"):
        validate_dataframe(invalid_df2)


def test_validate_dataframe_negative_rates_and_ratios(synthetic_train_df: pd.DataFrame) -> None:
    """Negative values in scanning rates, turnover speed, or void ratios must be caught."""
    for col in ["scannedLineItemsPerSecond", "valuePerSecond", "lineItemVoidsPerPosition"]:
        bad_df = synthetic_train_df.copy()
        bad_df.loc[0, col] = -0.5
        with pytest.raises(SchemaValidationError, match="contains .* negative value"):
            validate_dataframe(bad_df)


def test_validate_dataframe_invalid_target_values(synthetic_train_df: pd.DataFrame) -> None:
    """Non-binary or fractional target values must raise SchemaValidationError."""
    bad_df = synthetic_train_df.copy()
    bad_df.loc[0, "fraud"] = 2
    with pytest.raises(SchemaValidationError, match="must contain only binary values 0 or 1"):
        validate_dataframe(bad_df, require_target=True)

    bad_df2 = synthetic_train_df.copy()
    bad_df2["fraud"] = bad_df2["fraud"].astype(object)
    bad_df2.loc[0, "fraud"] = 0.5
    with pytest.raises(SchemaValidationError, match="contains non-integer numbers"):
        validate_dataframe(bad_df2, require_target=True)
