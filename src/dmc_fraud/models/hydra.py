"""The Hydra Ensemble Classifier combining XGBoost, Random Forest, and Logistic Regression."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from dmc_fraud.data.schema import TARGET_COLUMN

logger = logging.getLogger("dmc_fraud.models.hydra")


class HydraNotFittedError(NotFittedError, RuntimeError):
    """Raised when an operation requiring a fitted HydraClassifier is called before fitting."""


class HydraClassifier(BaseEstimator, ClassifierMixin):
    """The Hydra Soft-Voting Ensemble Classifier.

    Combines:
    1. XGBoost: High-capacity gradient boosting for capturing complex non-linear patterns.
    2. Random Forest: Bagging ensemble to suppress prediction variance and prevent overfitting.
    3. Logistic Regression (StandardScaler): Calibrated linear baseline for stable probability margins.

    Ensemble Soft-Voting weighting defaults to 2 : 1 : 0.5 (prioritizing XGBoost).
    """

    def __init__(
        self,
        xgb_params: dict[str, Any] | None = None,
        rf_params: dict[str, Any] | None = None,
        lr_params: dict[str, Any] | None = None,
        voting_weights: list[float] | None = None,
        danger_trust_levels: list[int] | None = None,
        filter_danger_on_train: bool = True,
        thresholds: dict[int, float] | None = None,
    ) -> None:
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        self.rf_params = rf_params or {
            "n_estimators": 300,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.lr_params = lr_params or {
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": 42,
        }
        self.voting_weights = voting_weights or [2.0, 1.0, 0.5]
        self.danger_trust_levels = danger_trust_levels or [1, 2]
        self.filter_danger_on_train = filter_danger_on_train
        self.thresholds = thresholds or {1: 0.85, 2: 0.75}

        # Fitted attributes
        self.voting_model_: VotingClassifier | None = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.feature_names_in_: list[str] | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | list,
    ) -> HydraClassifier:
        """Fit the Hydra soft-voting ensemble on training data.

        If filter_danger_on_train is True and 'trustLevel' is available in X,
        only high-risk danger levels (default: trustLevel <= 2) are utilized for model training.
        """
        y_arr = np.asarray(y).astype(int)

        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if TARGET_COLUMN in df.columns:
                df = df.drop(columns=[TARGET_COLUMN])

            if self.filter_danger_on_train and "trustLevel" in df.columns:
                mask = df["trustLevel"].isin(self.danger_trust_levels)
                y_filtered = y_arr[mask.values]
                if mask.sum() > 0 and np.sum(y_filtered == 1) > 0 and np.sum(y_filtered == 0) > 0:
                    df = df[mask]
                    y_arr = y_filtered
                else:
                    logger.warning(
                        "Danger trust level filter did not retain both fraud and clean classes; training on full dataset."
                    )

            self.feature_names_in_ = list(df.columns)
            X_train = df.values
        else:
            X_train = np.asarray(X)
            self.feature_names_in_ = [f"f_{i}" for i in range(X_train.shape[1])]

        if len(y_arr) == 0:
            raise ValueError("Training dataset contains 0 samples after filtering.")

        # Compute imbalance ratio for XGBoost scale_pos_weight
        n_neg = float(np.sum(y_arr == 0))
        n_pos = float(np.sum(y_arr == 1))
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Training dataset must contain both classes. Found {int(n_neg)} clean (0) and {int(n_pos)} fraud (1) samples."
            )
        ratio = n_neg / max(1.0, n_pos)

        # Create estimators
        xgb_cfg = dict(self.xgb_params)
        xgb_cfg["scale_pos_weight"] = ratio
        clf_xgb = XGBClassifier(**xgb_cfg)

        clf_rf = RandomForestClassifier(**self.rf_params)

        clf_lr = make_pipeline(StandardScaler(), LogisticRegression(**self.lr_params))

        self.voting_model_ = VotingClassifier(
            estimators=[("xgb", clf_xgb), ("rf", clf_rf), ("lr", clf_lr)],
            voting="soft",
            weights=self.voting_weights,
        )

        logger.info(
            "Fitting Hydra Ensemble on %d samples (%d clean, %d fraud). Ratio: %.2f",
            len(y_arr),
            int(n_neg),
            int(n_pos),
            ratio,
        )
        self.voting_model_.fit(X_train, y_arr)
        self.ensemble_ = self.voting_model_
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Estimate fraud class probabilities [P(clean), P(fraud)] for each transaction."""
        try:
            check_is_fitted(self, "ensemble_")
        except NotFittedError as exc:
            raise HydraNotFittedError(str(exc)) from exc

        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if TARGET_COLUMN in df.columns:
                df = df.drop(columns=[TARGET_COLUMN])
            if self.feature_names_in_:
                missing = [c for c in self.feature_names_in_ if c not in df.columns]
                if missing:
                    raise ValueError(f"Input DataFrame is missing required features: {missing}")
                X_mat = df[self.feature_names_in_].values
            else:
                X_mat = df.values
        else:
            X_mat = np.asarray(X)

        assert self.voting_model_ is not None
        return self.voting_model_.predict_proba(X_mat)

    def predict_with_probs(
        self,
        probs: np.ndarray,
        X: pd.DataFrame | np.ndarray,
        thresholds: Mapping[Any, float] | None = None,
    ) -> np.ndarray:
        """Apply segment decision thresholds directly to precomputed fraud probabilities.

        Parameters
        ----------
        probs : np.ndarray
            1D array of estimated fraud probabilities (shape: n_samples,).
        X : Union[pd.DataFrame, np.ndarray]
            Features matrix or DataFrame with 'trustLevel' column.
        thresholds : Optional[Mapping[Any, float]]
            Segment threshold mapping, e.g. {1: 0.85, 2: 0.75}. Defaults to model thresholds.

        Returns
        -------
        np.ndarray
            Binary prediction vector (0 or 1).
        """
        n_samples = len(probs)
        preds = np.zeros(n_samples, dtype=int)
        active_thresholds = thresholds or self.thresholds

        if isinstance(X, pd.DataFrame) and "trustLevel" in X.columns:
            trust_levels = X["trustLevel"].values
            for lvl_raw, thresh in active_thresholds.items():
                lvl = int(lvl_raw)
                mask = (trust_levels == lvl) & (probs >= thresh)
                preds[mask] = 1
        else:
            # Fallback if trustLevel column is absent: use global mean threshold
            default_t = float(np.mean(list(active_thresholds.values()))) if active_thresholds else 0.5
            preds[probs >= default_t] = 1

        return preds

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        thresholds: Mapping[Any, float] | None = None,
    ) -> np.ndarray:
        """Make binary fraud predictions applying dual-gate / multi-gate segment thresholds.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Features matrix or DataFrame.
        thresholds : Optional[Dict[Union[int, str], float]]
            Segment threshold mapping, e.g. {1: 0.85, 2: 0.75}. Defaults to model's thresholds.

        Returns
        -------
        np.ndarray
            Binary prediction vector (0 or 1).
        """
        probs = self.predict_proba(X)[:, 1]
        return self.predict_with_probs(probs, X, thresholds=thresholds)
