"""Configuration models and loader for DMC 2019 Fraud Detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CALIBRATED_THRESHOLDS: dict[int, float] = {1: 0.85, 2: 0.75}


class FeatureConfig(BaseModel):
    """Configuration for feature engineering transformations."""

    epsilon: float | None = Field(
        default=1e-6,
        ge=0.0,
        description="Epsilon smoothing denominator constant for ratio calculations",
    )


class CostMatrixConfig(BaseModel):
    """Cost matrix for the DMC 2019 economic profit objective."""

    profit_tp: float = Field(default=5.0, description="Euro reward per True Positive (+€5 caught fraud)")
    penalty_fp: float = Field(default=25.0, description="Euro penalty per False Positive (-€25 false accusation)")
    penalty_fn: float = Field(default=5.0, description="Euro penalty per False Negative (-€5 missed fraud)")
    reward_tn: float = Field(default=0.0, description="Euro reward per True Negative (€0 clean pass)")


class ModelHyperparameters(BaseModel):
    """Hyperparameters for the constituent models in the Hydra ensemble."""

    xgb_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "eval_metric": "logloss",
        }
    )
    rf_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    lr_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": 42,
        }
    )
    voting_weights: list[float] = Field(
        default_factory=lambda: [2.0, 1.0, 0.5],
        description="Voting weights for [XGBoost, Random Forest, Logistic Regression]",
    )
    danger_trust_levels: list[int] = Field(
        default_factory=lambda: [1, 2],
        description="Trust levels flagged as risky (trained and evaluated with dual-gate)",
    )


class CalibrationConfig(BaseModel):
    """Configuration for Out-of-Fold Dual-Gate threshold calibration."""

    cv_folds: int = Field(default=5, ge=2, description="Number of stratified folds for OOF calibration")
    threshold_min: float = Field(default=0.50, ge=0.0, le=1.0)
    threshold_max: float = Field(default=0.99, ge=0.0, le=1.0)
    threshold_step: float = Field(default=0.05, gt=0.0, le=0.5)
    random_state: int = Field(default=42)


class DMCConfig(BaseModel):
    """Unified configuration schema for DMC Fraud Detection pipeline."""

    costs: CostMatrixConfig = Field(default_factory=CostMatrixConfig)
    models: ModelHyperparameters = Field(default_factory=ModelHyperparameters)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    data_delimiter: str = Field(default="|", description="CSV field delimiter")
    version: str = Field(default="1.0.0")

    def to_yaml(self, path: str | Path) -> None:
        """Export configuration to YAML file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DMCConfig:
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> DMCConfig:
        """Load configuration from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def load_config(path: str | Path | None = None) -> DMCConfig:
    """Load configuration from a path or return default configuration."""
    if path is None:
        return DMCConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return DMCConfig.from_yaml(config_path)
    elif suffix == ".json":
        return DMCConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config extension '{suffix}'. Expected .yaml, .yml, or .json")
