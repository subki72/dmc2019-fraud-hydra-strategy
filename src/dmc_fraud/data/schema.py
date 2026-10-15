"""Data schema definitions, column constants, and DataFrame validation for DMC 2019."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

RAW_FEATURE_COLUMNS: list[str] = [
    "trustLevel",
    "totalScanTimeInSeconds",
    "grandTotal",
    "lineItemVoids",
    "scansWithoutRegistration",
    "quantityModifications",
    "scannedLineItemsPerSecond",
    "valuePerSecond",
    "lineItemVoidsPerPosition",
]

TARGET_COLUMN: str = "fraud"

ENGINEERED_FEATURE_COLUMNS: list[str] = [
    "scannedLineItemsTotal",
    "avgTimePerItem",
    "voidsPerTotalItem",
    "noRegPerItem",
    "pricePerItem",
    "log_grandTotal",
    "log_totalScanTime",
    "interaction_trust_time",
    "interaction_void_price",
]

ALL_FEATURE_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


class SchemaValidationError(ValueError):
    """Raised when an input DataFrame fails DMC 2019 schema validation."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class TransactionRecord(BaseModel):
    """Pydantic model representing a single DMC 2019 transaction record."""

    trustLevel: int = Field(ge=1, le=6, description="Customer trust level (1=lowest, 6=highest)")
    totalScanTimeInSeconds: float = Field(ge=0.0, description="Total duration of checkout scan session")
    grandTotal: float = Field(ge=0.0, description="Total monetary checkout value in Euros")
    lineItemVoids: int = Field(ge=0, description="Count of voided scanned items")
    scansWithoutRegistration: int = Field(ge=0, description="Count of attempted scans without item registration")
    quantityModifications: int = Field(ge=0, description="Count of manual item quantity modifications")
    scannedLineItemsPerSecond: float = Field(ge=0.0, description="Scanning speed in line items per second")
    valuePerSecond: float = Field(ge=0.0, description="Monetary turnover rate in Euros per second")
    lineItemVoidsPerPosition: float = Field(ge=0.0, description="Ratio of item voids per scanned position")
    fraud: int | None = Field(default=None, ge=0, le=1, description="Ground truth fraud label (0=clean, 1=fraud)")


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = False,
    strict_bounds: bool = True,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the DMC 2019 feature and target schemas.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    require_target : bool, default=False
        Whether the 'fraud' target column must be present and valid.
    strict_bounds : bool, default=True
        Whether to enforce physical bounds (e.g. trustLevel in 1..6, non-negative values).

    Returns
    -------
    pd.DataFrame
        Validated copy of the DataFrame with appropriate column types cast.

    Raises
    ------
    SchemaValidationError
        If missing columns, invalid data types, null values, or out-of-bound values are found.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise SchemaValidationError("Input must be a valid pandas DataFrame.")

    if df.empty:
        raise SchemaValidationError("Input DataFrame is empty.")

    errors: list[str] = []

    # Check required raw feature columns
    missing_features = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required raw feature columns: {missing_features}")

    # Check target column if required
    if require_target and TARGET_COLUMN not in df.columns:
        errors.append(f"Missing required target column: '{TARGET_COLUMN}'")

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed with {len(errors)} error(s): " + "; ".join(errors),
            errors=errors,
        )

    # Work on a copy to cast dtypes safely
    clean_df = df.copy()

    # Check for NaN / null values in required features
    null_counts = clean_df[RAW_FEATURE_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = [f"{col} ({count} nulls)" for col, count in cols_with_nulls.items()]
        errors.append(f"Null values detected in feature columns: {', '.join(null_details)}")

    if require_target and TARGET_COLUMN in clean_df.columns:
        target_nulls = clean_df[TARGET_COLUMN].isnull().sum()
        if target_nulls > 0:
            errors.append(f"Target column '{TARGET_COLUMN}' contains {target_nulls} null values.")

    # Cast types and enforce discrete integrity
    integer_cols = ["trustLevel", "lineItemVoids", "scansWithoutRegistration", "quantityModifications"]
    for col in integer_cols:
        try:
            numeric_s = pd.to_numeric(clean_df[col], errors="raise")
            if not (numeric_s % 1 == 0).all():
                errors.append(f"Column '{col}' must contain integer values, but non-integer numbers were detected.")
            clean_df[col] = numeric_s.astype(int)
        except (ValueError, TypeError) as e:
            errors.append(f"Column '{col}' must be integer: {e}")

    numeric_float_cols = [
        "totalScanTimeInSeconds",
        "grandTotal",
        "scannedLineItemsPerSecond",
        "valuePerSecond",
        "lineItemVoidsPerPosition",
    ]
    for col in numeric_float_cols:
        try:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="raise").astype(float)
        except (ValueError, TypeError) as e:
            errors.append(f"Column '{col}' must be numeric float: {e}")

    if strict_bounds and not errors:
        # Check trustLevel bounds [1, 6]
        invalid_trust = clean_df[~clean_df["trustLevel"].isin([1, 2, 3, 4, 5, 6])]
        if not invalid_trust.empty:
            errors.append(
                f"'trustLevel' must be in range [1..6], found invalid values: "
                f"{invalid_trust['trustLevel'].unique().tolist()}"
            )

        # Check non-negative constraints across all 8 numeric metrics
        non_negative_cols = [
            "totalScanTimeInSeconds",
            "grandTotal",
            "lineItemVoids",
            "scansWithoutRegistration",
            "quantityModifications",
            "scannedLineItemsPerSecond",
            "valuePerSecond",
            "lineItemVoidsPerPosition",
        ]
        for col in non_negative_cols:
            neg_count = (clean_df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' contains {neg_count} negative value(s).")

    if require_target and TARGET_COLUMN in clean_df.columns:
        try:
            target_numeric = pd.to_numeric(clean_df[TARGET_COLUMN], errors="raise")
            if not (target_numeric % 1 == 0).all():
                errors.append(f"Target column '{TARGET_COLUMN}' contains non-integer numbers.")
            clean_df[TARGET_COLUMN] = target_numeric.astype(int)
            invalid_targets = clean_df[~clean_df[TARGET_COLUMN].isin([0, 1])]
            if not invalid_targets.empty:
                errors.append(
                    f"Target column '{TARGET_COLUMN}' must contain only binary values 0 or 1, "
                    f"found: {invalid_targets[TARGET_COLUMN].unique().tolist()}"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Target column '{TARGET_COLUMN}' could not be cast to binary integer: {e}")

    if errors:
        raise SchemaValidationError(
            f"Schema validation failed with {len(errors)} error(s): " + "; ".join(errors),
            errors=errors,
        )

    return clean_df
