"""Unit tests for ModelBundle persistence and cryptographic tamper detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmc_fraud.data.schema import TARGET_COLUMN
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.bundle import IntegrityError, ModelBundle
from dmc_fraud.models.hydra import HydraClassifier


@pytest.fixture
def trained_bundle(synthetic_train_df: pd.DataFrame) -> ModelBundle:
    """Fixture providing a trained ModelBundle."""
    fe = FraudFeatureEngineer()
    X_train_adv = fe.fit_transform(synthetic_train_df)
    y_train = synthetic_train_df[TARGET_COLUMN].values

    model = HydraClassifier(
        xgb_params={"n_estimators": 5, "max_depth": 2, "eval_metric": "logloss", "random_state": 42},
        rf_params={"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1},
        thresholds={1: 0.82, 2: 0.72},
    )
    model.fit(X_train_adv, y_train)

    return ModelBundle(
        model=model,
        feature_engineer=fe,
        thresholds={1: 0.82, 2: 0.72},
        metadata={"version": "1.0.0", "author": "Test Author"},
    )


def test_bundle_save_load_roundtrip(
    trained_bundle: ModelBundle, synthetic_test_df: pd.DataFrame, tmp_path: Path
) -> None:
    """ModelBundle must serialize and deserialize cleanly, producing identical inference."""
    bundle_path = tmp_path / "model_bundle.joblib"
    saved_path = trained_bundle.save(bundle_path)

    assert saved_path.exists()
    checksum_path = bundle_path.with_suffix(".joblib.sha256")
    assert checksum_path.exists()

    # Load bundle
    loaded_bundle = ModelBundle.load(bundle_path, verify_checksum=True)

    assert loaded_bundle.thresholds == trained_bundle.thresholds
    assert loaded_bundle.metadata["version"] == "1.0.0"

    # Compare predictions
    preds_orig = trained_bundle.predict(synthetic_test_df)
    preds_loaded = loaded_bundle.predict(synthetic_test_df)
    np.testing.assert_array_equal(preds_orig, preds_loaded)

    # Compare probabilities
    probs_orig = trained_bundle.predict_proba(synthetic_test_df)
    probs_loaded = loaded_bundle.predict_proba(synthetic_test_df)
    np.testing.assert_allclose(probs_orig, probs_loaded)


def test_bundle_cryptographic_tamper_detection(trained_bundle: ModelBundle, tmp_path: Path) -> None:
    """Tampering with a saved model file must trigger an IntegrityError upon load."""
    bundle_path = tmp_path / "tamper_test.joblib"
    trained_bundle.save(bundle_path)

    # Corrupt a byte in the artifact file
    content = bytearray(bundle_path.read_bytes())
    content[-10] = (content[-10] + 1) % 256
    bundle_path.write_bytes(content)

    with pytest.raises(IntegrityError, match="Cryptographic integrity check failed"):
        ModelBundle.load(bundle_path, verify_checksum=True)


def test_bundle_evaluate(trained_bundle: ModelBundle, synthetic_test_with_target: pd.DataFrame) -> None:
    """Verify bundle evaluate method computes valid evaluation dictionary."""
    X_test = synthetic_test_with_target.drop(columns=[TARGET_COLUMN])
    y_test = synthetic_test_with_target[TARGET_COLUMN].values

    metrics = trained_bundle.evaluate(X_test, y_test)
    assert "profit" in metrics
    assert "true_positives" in metrics
    assert metrics["total_samples"] == len(synthetic_test_with_target)


def test_bundle_nonexistent_file_raises() -> None:
    """Loading a nonexistent bundle path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found at"):
        ModelBundle.load("nonexistent_bundle.joblib")


def test_bundle_predict_with_proba(trained_bundle: ModelBundle, synthetic_test_df: pd.DataFrame) -> None:
    """Verify single-pass predict_with_proba produces identical outputs to separate calls."""
    preds_single, probs_single = trained_bundle.predict_with_proba(synthetic_test_df)
    preds_sep = trained_bundle.predict(synthetic_test_df)
    probs_sep = trained_bundle.predict_proba(synthetic_test_df)[:, 1]

    np.testing.assert_array_equal(preds_single, preds_sep)
    np.testing.assert_allclose(probs_single, probs_sep)


def test_bundle_numpy_array_input(trained_bundle: ModelBundle, synthetic_test_with_target: pd.DataFrame) -> None:
    """Verify ModelBundle supports 2D numpy arrays across predict, predict_proba, and evaluate."""
    from dmc_fraud.data.schema import RAW_FEATURE_COLUMNS

    X_mat = synthetic_test_with_target[RAW_FEATURE_COLUMNS].values
    y_vec = synthetic_test_with_target[TARGET_COLUMN].values

    preds = trained_bundle.predict(X_mat, validate=True)
    assert len(preds) == len(X_mat)

    probs = trained_bundle.predict_proba(X_mat, validate=True)
    assert probs.shape == (len(X_mat), 2)

    metrics = trained_bundle.evaluate(X_mat, y_vec, validate=True)
    assert "profit" in metrics


def test_bundle_alt_sha256_suffix(trained_bundle: ModelBundle, tmp_path: Path) -> None:
    """Verify ModelBundle resolves companion checksum file with plain .sha256 extension."""
    from dmc_fraud.models.bundle import compute_file_sha256

    bundle_path = tmp_path / "alt_bundle.joblib"
    trained_bundle.save(bundle_path, write_checksum_file=False)

    # Write plain .sha256 companion (alt_bundle.sha256)
    alt_sha_path = bundle_path.with_suffix(".sha256")
    alt_sha_path.write_text(f"{compute_file_sha256(bundle_path)}  {bundle_path.name}\n")

    loaded = ModelBundle.load(bundle_path, verify_checksum=True)
    assert loaded.thresholds == trained_bundle.thresholds


def test_bundle_load_base_dir_confinement_success(trained_bundle: ModelBundle, tmp_path: Path) -> None:
    """Loading bundle within authorized base_dir succeeds."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bundle_path = models_dir / "confined.joblib"
    trained_bundle.save(bundle_path)

    loaded = ModelBundle.load(bundle_path, base_dir=tmp_path)
    assert loaded is not None


def test_bundle_load_base_dir_confinement_violation(trained_bundle: ModelBundle, tmp_path: Path) -> None:
    """Loading bundle outside authorized base_dir raises ValueError."""
    bundle_path = tmp_path / "outside.joblib"
    trained_bundle.save(bundle_path)

    other_dir = tmp_path / "other"
    other_dir.mkdir()

    with pytest.raises(ValueError, match="Path confinement violation"):
        ModelBundle.load(bundle_path, base_dir=other_dir)


def test_bundle_load_default_calibrated_thresholds(tmp_path: Path, trained_bundle: ModelBundle) -> None:
    """Loading bundle without thresholds field defaults to DEFAULT_CALIBRATED_THRESHOLDS."""
    import joblib

    from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS

    bundle_path = tmp_path / "no_thresh.joblib"
    trained_bundle.save(bundle_path)

    # Modify payload to drop thresholds
    payload = joblib.load(bundle_path)
    del payload["thresholds"]
    joblib.dump(payload, bundle_path)

    # Load with verify_checksum=False because content changed
    loaded = ModelBundle.load(bundle_path, verify_checksum=False)
    assert loaded.thresholds == DEFAULT_CALIBRATED_THRESHOLDS
