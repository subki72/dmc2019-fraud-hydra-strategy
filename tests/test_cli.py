"""Integration tests for the dmc-fraud CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dmc_fraud.cli.main import app
from dmc_fraud.config import DMCConfig

runner = CliRunner()


def test_cli_version_and_help() -> None:
    """CLI should print version and help with exit code 0."""
    res_help = runner.invoke(app, ["--help"])
    assert res_help.exit_code == 0
    assert "DMC 2019 Retail Fraud Detection" in res_help.stdout

    res_version = runner.invoke(app, ["--version"])
    assert res_version.exit_code == 0
    assert "1.0.0" in res_version.stdout


def test_cli_train_and_info(temp_data_dir: Path, tmp_path: Path) -> None:
    """Test 'dmc-fraud train' and 'dmc-fraud info' end-to-end."""
    train_csv = temp_data_dir / "train.csv"
    bundle_path = tmp_path / "model.joblib"

    # Fast configuration
    cfg = DMCConfig()
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "eval_metric": "logloss", "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}
    cfg_file = tmp_path / "fast_cfg.yaml"
    cfg.to_yaml(cfg_file)

    # 1. Train command with OOF calibration (2 folds for test speed)
    train_args = [
        "train",
        "--train-data",
        str(train_csv),
        "--output-bundle",
        str(bundle_path),
        "--folds",
        "2",
        "--config",
        str(cfg_file),
    ]
    res_train = runner.invoke(app, train_args)
    assert res_train.exit_code == 0, f"Train failed with: {res_train.stdout}"
    assert bundle_path.exists()
    assert bundle_path.with_suffix(".joblib.sha256").exists()
    assert "Model bundle saved successfully" in res_train.stdout

    # 2. Info command
    res_info = runner.invoke(app, ["info", "--model-bundle", str(bundle_path)])
    assert res_info.exit_code == 0
    assert "MODEL BUNDLE INFO" in res_info.stdout
    assert "Decision Thresholds" in res_info.stdout


def test_cli_predict_and_evaluate(temp_data_dir: Path, tmp_path: Path) -> None:
    """Test 'dmc-fraud predict' and 'dmc-fraud evaluate' commands end-to-end."""
    train_csv = temp_data_dir / "train.csv"
    test_csv = temp_data_dir / "test.csv"
    labels_csv = temp_data_dir / "realclass.csv"
    bundle_path = tmp_path / "model_predict.joblib"
    preds_output = tmp_path / "predictions.csv"
    eval_json = tmp_path / "metrics.json"

    # Train model fast with skip-calibration
    runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(train_csv),
            "--output-bundle",
            str(bundle_path),
            "--skip-calibration",
        ],
    )

    # 1. Predict command
    predict_args = [
        "predict",
        "--model-bundle",
        str(bundle_path),
        "--input-data",
        str(test_csv),
        "--output",
        str(preds_output),
        "--include-probabilities",
    ]
    res_predict = runner.invoke(app, predict_args)
    assert res_predict.exit_code == 0, f"Predict failed: {res_predict.stdout}"
    assert preds_output.exists()

    df_preds = pd.read_csv(preds_output)
    assert "fraud" in df_preds.columns
    assert "probability" in df_preds.columns
    assert len(df_preds) == 60

    # 2. Evaluate command
    eval_args = [
        "evaluate",
        "--model-bundle",
        str(bundle_path),
        "--test-data",
        str(test_csv),
        "--labels",
        str(labels_csv),
        "--output",
        str(eval_json),
    ]
    res_eval = runner.invoke(app, eval_args)
    assert res_eval.exit_code == 0, f"Evaluate failed: {res_eval.stdout}"
    assert "DMC 2019 EVALUATION REPORT" in res_eval.stdout
    assert eval_json.exists()

    metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    assert "profit" in metrics
    assert "precision" in metrics


def test_cli_calibrate_command(temp_data_dir: Path, tmp_path: Path) -> None:
    """Test 'dmc-fraud calibrate' standalone command."""
    train_csv = temp_data_dir / "train.csv"

    # Fast config
    cfg = DMCConfig()
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "eval_metric": "logloss", "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}
    cfg_file = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_file)

    res_calib = runner.invoke(
        app,
        [
            "calibrate",
            "--train-data",
            str(train_csv),
            "--folds",
            "2",
            "--config",
            str(cfg_file),
        ],
    )
    assert res_calib.exit_code == 0
    assert "CALIBRATION RESULTS" in res_calib.stdout
    assert "Out-of-Fold Economic Profit" in res_calib.stdout


def test_cli_error_handling() -> None:
    """CLI should exit with code 1 and error message on invalid input."""
    res = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            "nonexistent.joblib",
            "--input-data",
            "dummy.csv",
            "--output",
            "out.csv",
        ],
    )
    assert res.exit_code == 1
    assert "ERROR:" in res.stderr or "ERROR:" in res.stdout


def test_cli_predict_chunksize(temp_data_dir: Path, tmp_path: Path) -> None:
    """Test 'dmc-fraud predict' streaming with --chunksize and --include-probabilities."""
    train_csv = temp_data_dir / "train.csv"
    test_csv = temp_data_dir / "test.csv"
    bundle_path = tmp_path / "model_chunk.joblib"
    preds_output = tmp_path / "stream_preds.csv"

    # Train model fast with skip-calibration
    runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(train_csv),
            "--output-bundle",
            str(bundle_path),
            "--skip-calibration",
        ],
    )

    res_predict = runner.invoke(
        app,
        [
            "predict",
            "--model-bundle",
            str(bundle_path),
            "--input-data",
            str(test_csv),
            "--output",
            str(preds_output),
            "--include-probabilities",
            "--chunksize",
            "20",
        ],
    )
    assert res_predict.exit_code == 0, f"Streaming predict failed: {res_predict.stdout}"
    assert "Streaming predictions in chunks of 20" in res_predict.stdout
    assert preds_output.exists()

    df_preds = pd.read_csv(preds_output)
    assert len(df_preds) == 60
    assert "fraud" in df_preds.columns
    assert "probability" in df_preds.columns


def test_cli_train_with_delimiter(tmp_path: Path, synthetic_train_df: pd.DataFrame) -> None:
    """Test 'dmc-fraud train' with explicit --delimiter option."""
    comma_csv = tmp_path / "train_comma.csv"
    synthetic_train_df.to_csv(comma_csv, sep=",", index=False)
    bundle_path = tmp_path / "model_comma.joblib"

    res = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(comma_csv),
            "--output-bundle",
            str(bundle_path),
            "--delimiter",
            ",",
            "--skip-calibration",
        ],
    )
    assert res.exit_code == 0, f"Train with delimiter failed: {res.stdout}"
    assert bundle_path.exists()


def test_cli_calibrate_nonexistent_output_bundle_notice(temp_data_dir: Path, tmp_path: Path) -> None:
    """Verify notice is logged when --output-bundle does not exist to update."""
    train_csv = temp_data_dir / "train.csv"
    nonexistent_bundle = tmp_path / "does_not_exist.joblib"

    res = runner.invoke(
        app,
        [
            "calibrate",
            "--train-data",
            str(train_csv),
            "--folds",
            "2",
            "--output-bundle",
            str(nonexistent_bundle),
        ],
    )
    assert res.exit_code == 0
    assert "Notice: Output bundle" in res.stdout


def test_cli_global_quiet_flag(temp_data_dir: Path, tmp_path: Path) -> None:
    """Global --quiet / -q suppresses banners, progress messages, and tables."""
    train_csv = temp_data_dir / "train.csv"
    test_csv = temp_data_dir / "test.csv"
    labels_csv = temp_data_dir / "realclass.csv"
    bundle_path = tmp_path / "quiet_model.joblib"
    eval_json = tmp_path / "quiet_metrics.json"

    # 1. Train in quiet mode
    res_train = runner.invoke(
        app,
        [
            "-q",
            "train",
            "--train-data",
            str(train_csv),
            "--output-bundle",
            str(bundle_path),
            "--skip-calibration",
        ],
    )
    assert res_train.exit_code == 0
    assert bundle_path.exists()
    assert res_train.stdout.strip() == ""

    # 2. Evaluate in quiet mode
    res_eval = runner.invoke(
        app,
        [
            "--quiet",
            "evaluate",
            "--model-bundle",
            str(bundle_path),
            "--test-data",
            str(test_csv),
            "--labels",
            str(labels_csv),
            "--output",
            str(eval_json),
        ],
    )
    assert res_eval.exit_code == 0
    assert eval_json.exists()
    assert "DMC 2019 EVALUATION REPORT" not in res_eval.stdout
    assert res_eval.stdout.strip() == ""


def test_cli_train_wires_config_epsilon(temp_data_dir: Path, tmp_path: Path) -> None:
    """Verify train command passes config features.epsilon to FraudFeatureEngineer."""
    from dmc_fraud.models.bundle import ModelBundle

    train_csv = temp_data_dir / "train.csv"
    bundle_path = tmp_path / "eps_model.joblib"
    cfg = DMCConfig()
    cfg.features.epsilon = 1e-3
    cfg.models.xgb_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42}
    cfg.models.rf_params = {"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1}
    cfg_file = tmp_path / "eps_cfg.yaml"
    cfg.to_yaml(cfg_file)

    res = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(train_csv),
            "--output-bundle",
            str(bundle_path),
            "--config",
            str(cfg_file),
            "--skip-calibration",
        ],
    )
    assert res.exit_code == 0
    bundle = ModelBundle.load(bundle_path)
    assert bundle.feature_engineer.epsilon == 1e-3
