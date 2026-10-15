"""Stratified K-Fold Out-of-Fold Dual-Gate threshold calibrator for DMC 2019."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS, DMCConfig
from dmc_fraud.evaluation.metrics import calculate_dmc_profit, evaluate_predictions
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.hydra import HydraClassifier

logger = logging.getLogger("dmc_fraud.evaluation.calibrator")


@dataclass
class CalibrationResult:
    """Encapsulates the optimal Out-of-Fold threshold calibration results."""

    best_t1: float
    best_t2: float
    best_profit: float
    oof_metrics: dict[str, Any]
    threshold_history: list[dict[str, Any]] = field(default_factory=list)
    oof_probabilities: np.ndarray | None = None
    oof_true_labels: np.ndarray | None = None


class DualGateCalibrator:
    """Scientific Out-of-Fold (OOF) Calibrator for asymmetric cost matrix optimization.

    Eliminates data leakage by deriving optimal decision boundaries solely from
    cross-validation folds within the training dataset, preserving the test set
    strictly as an unseen holdout benchmark.
    """

    def __init__(
        self,
        config: DMCConfig | None = None,
        cv_folds: int = 5,
        threshold_min: float = 0.50,
        threshold_max: float = 0.99,
        threshold_step: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.config = config or DMCConfig()
        self.cv_folds = cv_folds if cv_folds is not None else self.config.calibration.cv_folds
        self.threshold_min = threshold_min if threshold_min is not None else self.config.calibration.threshold_min
        self.threshold_max = threshold_max if threshold_max is not None else self.config.calibration.threshold_max
        self.threshold_step = threshold_step if threshold_step is not None else self.config.calibration.threshold_step
        self.random_state = random_state if random_state is not None else self.config.calibration.random_state

        if self.threshold_min > self.threshold_max:
            raise ValueError(f"threshold_min ({self.threshold_min}) cannot exceed threshold_max ({self.threshold_max})")
        if self.threshold_step <= 0:
            raise ValueError(f"threshold_step ({self.threshold_step}) must be positive")

    def calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray | list,
    ) -> CalibrationResult:
        """Execute Stratified K-Fold Out-of-Fold probability estimation and dual-gate grid search.

        Parameters
        ----------
        X_train : pd.DataFrame
            Raw or engineered training features containing 'trustLevel'.
        y_train : Union[pd.Series, np.ndarray, list]
            Training binary ground-truth labels.

        Returns
        -------
        CalibrationResult
            Calibrated thresholds, OOF profit, and evaluation metrics.
        """
        y_arr = np.asarray(y_train).astype(int)
        n_samples = len(X_train)

        if n_samples != len(y_arr):
            raise ValueError(f"X_train length ({n_samples}) does not match y_train length ({len(y_arr)})")

        if "trustLevel" not in X_train.columns:
            raise ValueError("Input DataFrame must include 'trustLevel' column for dual-gate calibration.")

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Cannot calibrate: dataset must contain both classes (found {n_pos} fraud, {n_neg} clean samples)."
            )

        effective_folds = min(self.cv_folds, n_pos, n_neg)
        if effective_folds < 2:
            raise ValueError(
                f"Cannot calibrate: minority class has only {min(n_pos, n_neg)} sample(s), requiring at least 2 for cross-validation."
            )
        if effective_folds < self.cv_folds:
            logger.warning(
                "Requested %d folds exceeds minority class count (%d). Dynamically adjusting cv_folds to %d.",
                self.cv_folds,
                min(n_pos, n_neg),
                effective_folds,
            )

        logger.info(
            "Starting %d-fold Stratified K-Fold Out-of-Fold calibration on %d training samples...",
            effective_folds,
            n_samples,
        )

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)
        oof_probs = np.zeros(n_samples, dtype=float)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_arr), 1):
            X_tr_raw = X_train.iloc[train_idx].copy()
            y_tr = y_arr[train_idx]
            X_val_raw = X_train.iloc[val_idx].copy()

            # Ensure features are engineered within fold (or pass through if already transformed)
            need_transform = "avgTimePerItem" not in X_tr_raw.columns
            if need_transform:
                fe_fold = FraudFeatureEngineer(epsilon=self.config.features.epsilon)
                X_tr = fe_fold.fit_transform(X_tr_raw)
                X_val = fe_fold.transform(X_val_raw)
            else:
                X_tr = X_tr_raw
                X_val = X_val_raw

            model = HydraClassifier(
                xgb_params=self.config.models.xgb_params,
                rf_params=self.config.models.rf_params,
                lr_params=self.config.models.lr_params,
                voting_weights=self.config.models.voting_weights,
                danger_trust_levels=self.config.models.danger_trust_levels,
            )
            model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs
            logger.info(
                "Fold %d/%d completed. Validation mean fraud probability: %.4f",
                fold,
                effective_folds,
                float(np.mean(val_probs)),
            )

        # Execute independent threshold optimization across disjoint subsets (O(2K) complexity)
        logger.info("Executing independent threshold optimization on out-of-fold probability distributions...")
        # Ensure threshold_max is inclusive with high-precision stepping
        threshold_range = np.round(
            np.arange(self.threshold_min, self.threshold_max + (self.threshold_step / 2.0), self.threshold_step),
            4,
        )
        threshold_range = threshold_range[threshold_range <= round(self.threshold_max, 4)]

        trust_levels = X_train["trustLevel"].values
        has_l1 = bool((trust_levels == 1).any())
        has_l2 = bool((trust_levels == 2).any())

        default_t1 = DEFAULT_CALIBRATED_THRESHOLDS[1]
        default_t2 = DEFAULT_CALIBRATED_THRESHOLDS[2]

        range_t1 = threshold_range if has_l1 else [default_t1]
        range_t2 = threshold_range if has_l2 else [default_t2]

        # 1. Optimize Level 1 threshold independently
        best_p1 = -float("inf")
        best_t1 = default_t1
        l1_profits: dict[float, float] = {}
        if has_l1:
            mask1 = trust_levels == 1
            y_l1 = y_arr[mask1]
            probs_l1 = oof_probs[mask1]
            for t1 in range_t1:
                preds_l1 = (probs_l1 >= t1).astype(int)
                p1 = calculate_dmc_profit(
                    y_l1,
                    preds_l1,
                    tp_reward=self.config.costs.profit_tp,
                    fp_penalty=self.config.costs.penalty_fp,
                    fn_penalty=self.config.costs.penalty_fn,
                    tn_reward=self.config.costs.reward_tn,
                )
                l1_profits[float(t1)] = p1
                # Conservative tie-breaking: prefer higher threshold to protect against -€25 FP penalty
                if p1 > best_p1 or (p1 == best_p1 and t1 > best_t1):
                    best_p1 = p1
                    best_t1 = float(t1)
        else:
            best_p1 = 0.0
            l1_profits[default_t1] = 0.0

        # 2. Optimize Level 2 threshold independently
        best_p2 = -float("inf")
        best_t2 = default_t2
        l2_profits: dict[float, float] = {}
        if has_l2:
            mask2 = trust_levels == 2
            y_l2 = y_arr[mask2]
            probs_l2 = oof_probs[mask2]
            for t2 in range_t2:
                preds_l2 = (probs_l2 >= t2).astype(int)
                p2 = calculate_dmc_profit(
                    y_l2,
                    preds_l2,
                    tp_reward=self.config.costs.profit_tp,
                    fp_penalty=self.config.costs.penalty_fp,
                    fn_penalty=self.config.costs.penalty_fn,
                    tn_reward=self.config.costs.reward_tn,
                )
                l2_profits[float(t2)] = p2
                # Conservative tie-breaking: prefer higher threshold to protect against -€25 FP penalty
                if p2 > best_p2 or (p2 == best_p2 and t2 > best_t2):
                    best_p2 = p2
                    best_t2 = float(t2)
        else:
            best_p2 = 0.0
            l2_profits[default_t2] = 0.0

        # 3. Base profit from remaining trust levels (Level >= 3: all predictions are 0)
        mask_rest = (trust_levels != 1) & (trust_levels != 2)
        if mask_rest.any():
            y_rest = y_arr[mask_rest]
            preds_rest = np.zeros(len(y_rest), dtype=int)
            profit_rest = calculate_dmc_profit(
                y_rest,
                preds_rest,
                tp_reward=self.config.costs.profit_tp,
                fp_penalty=self.config.costs.penalty_fp,
                fn_penalty=self.config.costs.penalty_fn,
                tn_reward=self.config.costs.reward_tn,
            )
        else:
            profit_rest = 0.0

        # Compute final OOF metrics using the optimal thresholds
        final_oof_preds = np.zeros(n_samples, dtype=int)
        final_oof_preds[(trust_levels == 1) & (oof_probs >= best_t1)] = 1
        final_oof_preds[(trust_levels == 2) & (oof_probs >= best_t2)] = 1

        oof_metrics = evaluate_predictions(y_arr, final_oof_preds, costs=self.config.costs)
        best_profit = float(oof_metrics["profit"])

        # Record search evaluations in threshold_history (O(2K) complexity)
        history: list[dict[str, Any]] = []
        for t1 in range_t1:
            total_p = l1_profits[float(t1)] + best_p2 + profit_rest
            history.append(
                {
                    "t1": round(float(t1), 4),
                    "t2": round(float(best_t2), 4),
                    "profit": float(total_p),
                }
            )
        for t2 in range_t2:
            if float(t2) == float(best_t2):
                continue
            total_p = best_p1 + l2_profits[float(t2)] + profit_rest
            history.append(
                {
                    "t1": round(float(best_t1), 4),
                    "t2": round(float(t2), 4),
                    "profit": float(total_p),
                }
            )

        logger.info(
            "Calibration complete! Best OOF Profit: € {:,.2f} | Optimal T1: {:.2f} | Optimal T2: {:.2f}",
            best_profit,
            best_t1,
            best_t2,
        )

        return CalibrationResult(
            best_t1=round(best_t1, 4),
            best_t2=round(best_t2, 4),
            best_profit=best_profit,
            oof_metrics=oof_metrics,
            threshold_history=history,
            oof_probabilities=oof_probs,
            oof_true_labels=y_arr,
        )
