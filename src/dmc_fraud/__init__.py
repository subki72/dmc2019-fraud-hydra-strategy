"""DMC 2019 Retail Fraud Detection Package (dmc-fraud).

Production-grade modular library and CLI implementing the Hydra Soft-Voting Ensemble
with Out-of-Fold Dual-Gate threshold calibration.
"""

from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS, DMCConfig, FeatureConfig, load_config
from dmc_fraud.data.loader import load_data
from dmc_fraud.data.schema import (
    ALL_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)
from dmc_fraud.evaluation.calibrator import CalibrationResult, DualGateCalibrator
from dmc_fraud.evaluation.metrics import calculate_dmc_profit, evaluate_predictions
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.bundle import IntegrityError, ModelBundle
from dmc_fraud.models.hydra import HydraClassifier

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_CALIBRATED_THRESHOLDS",
    "DMCConfig",
    "FeatureConfig",
    "load_config",
    "RAW_FEATURE_COLUMNS",
    "ENGINEERED_FEATURE_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "validate_dataframe",
    "load_data",
    "FraudFeatureEngineer",
    "HydraClassifier",
    "ModelBundle",
    "IntegrityError",
    "calculate_dmc_profit",
    "evaluate_predictions",
    "DualGateCalibrator",
    "CalibrationResult",
    "__version__",
]
