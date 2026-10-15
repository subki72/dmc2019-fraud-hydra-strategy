"""Unified Model Artifact Bundle with cryptographic integrity verification."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS, DMCConfig
from dmc_fraud.data.schema import RAW_FEATURE_COLUMNS, validate_dataframe
from dmc_fraud.evaluation.metrics import evaluate_predictions
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.hydra import HydraClassifier

logger = logging.getLogger("dmc_fraud.models.bundle")


class IntegrityError(ValueError):
    """Raised when model artifact checksum verification fails."""


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelBundle:
    """Unified container encapsulating the complete fraud detection system.

    Includes:
    - Trained Hydra ensemble classifier
    - Calibrated feature engineering transformer
    - Optimal decision thresholds per customer risk segment
    - Schema specifications and metadata (version, git hash, training profit)
    - Cryptographic SHA-256 verification against tampering or silent corruption
    """

    def __init__(
        self,
        model: HydraClassifier,
        feature_engineer: FraudFeatureEngineer,
        thresholds: dict[int, float],
        metadata: dict[str, Any] | None = None,
        raw_columns: list[str] | None = None,
        config: DMCConfig | None = None,
    ) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.thresholds = thresholds
        self.raw_columns = raw_columns or list(RAW_FEATURE_COLUMNS)
        self.config = config or DMCConfig()
        self.metadata = metadata or {
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": "DMC 2019 Retail Fraud Hydra Model Bundle",
        }

    def save(self, path: str | Path, write_checksum_file: bool = True) -> Path:
        """Serialize model bundle to a compressed joblib archive and generate SHA256 checksum.

        Parameters
        ----------
        path : Union[str, Path]
            Destination path (e.g. 'models/dmc_hydra.joblib').
        write_checksum_file : bool, default=True
            Whether to write a companion '<path>.sha256' checksum file.

        Returns
        -------
        Path
            Path to the saved artifact bundle.
        """
        artifact_path = Path(path).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
            "raw_columns": self.raw_columns,
            "config": self.config.model_dump(),
        }

        logger.info("Serializing model bundle to: %s", artifact_path)
        joblib.dump(payload, artifact_path, compress=3)

        sha256_hash = compute_file_sha256(artifact_path)
        logger.info("Artifact SHA-256: %s", sha256_hash)

        if write_checksum_file:
            checksum_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
            checksum_path.write_text(f"{sha256_hash}  {artifact_path.name}\n", encoding="utf-8")
            logger.info("Checksum written to: %s", checksum_path)

        return artifact_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        verify_checksum: bool = True,
        base_dir: Path | str | None = None,
    ) -> ModelBundle:
        """Load and deserialize a model bundle with optional cryptographic verification.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the serialized joblib artifact.
        verify_checksum : bool, default=True
            Whether to verify integrity using companion '.sha256' file if present.
        base_dir : Optional[Union[Path, str]], default=None
            If provided, enforces path confinement to ensure the artifact resides within base_dir.

        Returns
        -------
        ModelBundle
            Reconstructed bundle instance ready for inference.

        Raises
        ------
        FileNotFoundError
            If artifact file is not found.
        ValueError
            If path confinement fails or bundle format is invalid.
        IntegrityError
            If checksum verification fails.
        """
        artifact_path = Path(path).resolve()
        if base_dir is not None:
            base_path = Path(base_dir).resolve()
            try:
                artifact_path.relative_to(base_path)
            except ValueError as exc:
                raise ValueError(
                    f"Path confinement violation: artifact path '{artifact_path}' is not within base directory '{base_path}'"
                ) from exc

        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact bundle not found at: {artifact_path}")

        if verify_checksum:
            checksum_path = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
            if not checksum_path.is_file():
                alt_checksum = artifact_path.with_suffix(".sha256")
                if alt_checksum.is_file():
                    checksum_path = alt_checksum
            if checksum_path.is_file():
                expected_sha = checksum_path.read_text(encoding="utf-8").strip().split()[0]
                actual_sha = compute_file_sha256(artifact_path)
                if expected_sha.lower() != actual_sha.lower():
                    raise IntegrityError(
                        f"Cryptographic integrity check failed! Expected SHA-256 '{expected_sha}', "
                        f"but got '{actual_sha}'. Artifact file may be tampered with or corrupted."
                    )
                logger.info("SHA-256 integrity check verified successfully (%s)", actual_sha[:12])

        logger.info("Deserializing model bundle from: %s", artifact_path)
        payload = joblib.load(artifact_path)

        if not isinstance(payload, dict) or "model" not in payload or "feature_engineer" not in payload:
            raise ValueError(f"Invalid model bundle format in {artifact_path}")

        cfg_dict = payload.get("config")
        config = DMCConfig(**cfg_dict) if cfg_dict else DMCConfig()

        raw_thresholds = payload.get("thresholds", DEFAULT_CALIBRATED_THRESHOLDS)
        thresholds = {int(k): float(v) for k, v in raw_thresholds.items()}

        return cls(
            model=payload["model"],
            feature_engineer=payload["feature_engineer"],
            thresholds=thresholds,
            metadata=payload.get("metadata", {}),
            raw_columns=payload.get("raw_columns", list(RAW_FEATURE_COLUMNS)),
            config=config,
        )

    def _prepare_input(self, X: pd.DataFrame | np.ndarray, validate: bool) -> pd.DataFrame:
        """Convert input array or DataFrame and validate schema if requested."""
        if isinstance(X, np.ndarray):
            if X.shape[1] == len(self.raw_columns):
                df = pd.DataFrame(X, columns=self.raw_columns)
            else:
                raise ValueError(
                    f"Array input dimension {X.shape[1]} does not match expected {len(self.raw_columns)} raw features."
                )
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

        return validate_dataframe(df, require_target=False) if validate else df

    def predict_proba(self, X: pd.DataFrame | np.ndarray, validate: bool = True) -> np.ndarray:
        """Compute fraud probabilities from raw transaction data."""
        df = self._prepare_input(X, validate=validate)
        X_adv = self.feature_engineer.transform(df)
        return self.model.predict_proba(X_adv)

    def predict(self, X: pd.DataFrame | np.ndarray, validate: bool = True) -> np.ndarray:
        """Generate binary fraud predictions using the bundle's calibrated dual-gate thresholds."""
        df = self._prepare_input(X, validate=validate)
        X_adv = self.feature_engineer.transform(df)
        return self.model.predict(X_adv, thresholds=self.thresholds)

    def predict_with_proba(self, X: pd.DataFrame | np.ndarray, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Compute both binary predictions and fraud probabilities in a single unified pass.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (binary_predictions, fraud_probabilities)
        """
        df = self._prepare_input(X, validate=validate)
        X_adv = self.feature_engineer.transform(df)
        probs = self.model.predict_proba(X_adv)[:, 1]
        preds = self.model.predict_with_probs(probs, X_adv, thresholds=self.thresholds)
        return preds, probs

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | list,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Evaluate predictions on a labeled dataset against the DMC 2019 profit matrix."""
        preds = self.predict(X, validate=validate)
        return evaluate_predictions(y, preds, costs=self.config.costs)
