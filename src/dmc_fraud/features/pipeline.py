"""Scikit-Learn compatible transformer for systematic DMC 2019 feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dmc_fraud.data.schema import ALL_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMN


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible Transformer that constructs the 9 systematic interaction

    and behavioral features for DMC 2019 fraud detection.

    Engineered features:
    1. scannedLineItemsTotal: Total items scanned (scannedLineItemsPerSecond * totalScanTimeInSeconds)
    2. avgTimePerItem: Average seconds per scanned item
    3. voidsPerTotalItem: Ratio of voided items to total items
    4. noRegPerItem: Ratio of scans without registration to total items
    5. pricePerItem: Average price in Euros per scanned item
    6. log_grandTotal: Logarithmic scale of total checkout amount
    7. log_totalScanTime: Logarithmic scale of scan time in seconds
    8. interaction_trust_time: Interaction between trust level and log scan time
    9. interaction_void_price: Interaction between item voids and price per item
    """

    def __init__(self, epsilon: float | None = 1e-6, copy: bool = True) -> None:
        self.epsilon = epsilon
        self.copy = copy
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any | None = None) -> FraudFeatureEngineer:
        """Fit transformer by verifying input columns and recording schema dimensions."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = len(self.feature_names_in_)
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Engineer 9 behavioral interaction features from raw input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Raw transaction features.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with both raw and engineered features.
        """
        if isinstance(X, np.ndarray):
            if len(RAW_FEATURE_COLUMNS) == X.shape[1]:
                df = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)
            else:
                raise ValueError(
                    f"Array input dimension {X.shape[1]} does not match expected {len(RAW_FEATURE_COLUMNS)} raw features."
                )
        elif isinstance(X, pd.DataFrame):
            df = X.copy() if self.copy else X
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

        # Ensure all required raw feature columns exist
        missing = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Cannot transform DataFrame. Missing required raw columns: {missing}")

        eps = self.epsilon

        # 1. scannedLineItemsTotal (reconstructed integer total items)
        raw_product = df["scannedLineItemsPerSecond"] * df["totalScanTimeInSeconds"]
        clean_product = raw_product.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        total_items = clean_product.round().astype(int)
        df["scannedLineItemsTotal"] = total_items

        # Safe conditional division preventing artificial extreme outliers when scannedLineItemsTotal == 0
        is_zero_items = df["scannedLineItemsTotal"] == 0
        denom = np.where(
            is_zero_items, 1.0, df["scannedLineItemsTotal"].astype(float) + (eps if eps is not None else 0.0)
        )

        # 2-5. Ratio-based behavioral features
        df["avgTimePerItem"] = np.where(is_zero_items, 0.0, df["totalScanTimeInSeconds"] / denom)
        df["voidsPerTotalItem"] = np.where(is_zero_items, 0.0, df["lineItemVoids"] / denom)
        df["noRegPerItem"] = np.where(is_zero_items, 0.0, df["scansWithoutRegistration"] / denom)
        df["pricePerItem"] = np.where(is_zero_items, 0.0, df["grandTotal"] / denom)

        # 6-7. Non-linear log transforms
        df["log_grandTotal"] = np.log1p(np.maximum(0.0, df["grandTotal"]))
        df["log_totalScanTime"] = np.log1p(np.maximum(0.0, df["totalScanTimeInSeconds"]))

        # 8-9. Cross-feature interaction terms
        df["interaction_trust_time"] = df["trustLevel"] * df["log_totalScanTime"]
        df["interaction_void_price"] = df["lineItemVoids"] * df["pricePerItem"]

        # Safeguard numerical anomalies: Inf and NaN replaced with 0.0
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Retain target column if present, else return feature columns
        ordered_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
        if TARGET_COLUMN in df.columns and TARGET_COLUMN not in ordered_cols:
            ordered_cols.append(TARGET_COLUMN)

        return df[ordered_cols]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the names of all features produced after transformation as ndarray."""
        return np.asarray(ALL_FEATURE_COLUMNS, dtype=object)
