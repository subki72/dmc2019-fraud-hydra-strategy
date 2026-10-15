"""Ergonomic Multi-Command CLI for DMC 2019 Retail Fraud Detection."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS, CostMatrixConfig, DMCConfig, load_config
from dmc_fraud.data.loader import load_data
from dmc_fraud.data.schema import TARGET_COLUMN
from dmc_fraud.evaluation.calibrator import CalibrationResult, DualGateCalibrator
from dmc_fraud.features.pipeline import FraudFeatureEngineer
from dmc_fraud.models.bundle import ModelBundle
from dmc_fraud.models.hydra import HydraClassifier

# Setup leveled logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dmc_fraud.cli")

app = typer.Typer(
    name="dmc-fraud",
    help="DMC 2019 Retail Fraud Detection - Hydra Strategy CLI & MLOps Toolkit",
    add_completion=False,
)

state: dict[str, Any] = {"quiet": False}


def _echo(msg: str, err: bool = False, color: str | None = None) -> None:
    """Print message to console unless global quiet flag is active."""
    if state.get("quiet", False) and not err:
        return
    if color:
        typer.secho(msg, fg=color, err=err)
    else:
        typer.echo(msg, err=err)


def _render_evaluation_table(metrics: dict[str, Any], costs: CostMatrixConfig) -> None:
    """Render the evaluation report with metrics and economic profit."""
    if state.get("quiet", False):
        return

    _echo("\n================ DMC 2019 EVALUATION REPORT ================")
    _echo(f"Total Economic Profit:     € {metrics['profit']:,.2f}")
    _echo(f"Profit Efficiency:         {metrics['profit_efficiency']:.2%}")
    _echo("------------------------------------------------------------")
    _echo(
        f"Caught Frauds (TP):        {metrics['true_positives']} (+€ {metrics['true_positives'] * costs.profit_tp:,.2f})"
    )
    _echo(
        f"False Accusations (FP):    {metrics['false_positives']} (-€ {metrics['false_positives'] * costs.penalty_fp:,.2f})"
    )
    _echo(
        f"Missed Frauds (FN):        {metrics['false_negatives']} (-€ {metrics['false_negatives'] * costs.penalty_fn:,.2f})"
    )
    if costs.reward_tn > 0:
        _echo(
            f"Clean Customers (TN):      {metrics['true_negatives']} (+€ {metrics['true_negatives'] * costs.reward_tn:,.2f})"
        )
    else:
        _echo(f"Clean Customers (TN):      {metrics['true_negatives']}")
    _echo("------------------------------------------------------------")
    _echo(f"Precision:                 {metrics['precision']:.4f}")
    _echo(f"Recall:                    {metrics['recall']:.4f}")
    _echo(f"F1-Score:                  {metrics['f1_score']:.4f}")
    _echo(f"Accuracy:                  {metrics['accuracy']:.4f}")
    _echo("============================================================\n")


def _render_calibration_table(res: CalibrationResult) -> None:
    """Render Out-of-Fold calibration results."""
    if state.get("quiet", False):
        return

    _echo("\n--- CALIBRATION RESULTS ---")
    _echo(f"Optimal Threshold (Trust Level 1): {res.best_t1:.4f}")
    _echo(f"Optimal Threshold (Trust Level 2): {res.best_t2:.4f}")
    _echo(f"Out-of-Fold Economic Profit:      € {res.best_profit:,.2f}")
    _echo(f"True Positives (Caught Fraud):     {res.oof_metrics['true_positives']}")
    _echo(f"False Positives (Accusations):    {res.oof_metrics['false_positives']}")
    _echo(f"False Negatives (Missed Fraud):   {res.oof_metrics['false_negatives']}")
    _echo(f"Precision:                        {res.oof_metrics['precision']:.4f}")
    _echo(f"Recall:                           {res.oof_metrics['recall']:.4f}")


def _render_bundle_info_table(bundle: ModelBundle) -> None:
    """Render model bundle metadata and structure."""
    if state.get("quiet", False):
        return

    _echo("\n--- MODEL BUNDLE INFO ---")
    _echo(f"Version:             {bundle.metadata.get('version', 'unknown')}")
    _echo(f"Created At:          {bundle.metadata.get('created_at', 'unknown')}")
    _echo(f"OOF Training Profit: € {bundle.metadata.get('oof_profit', 0.0):,.2f}")
    _echo(f"Decision Thresholds: {bundle.thresholds}")
    _echo(f"Expected Features:   {len(bundle.raw_columns)} raw features")
    if bundle.model.voting_model_:
        _echo(f"Ensemble Estimators: {[name for name, _ in bundle.model.voting_model_.estimators]}")
        _echo(f"Voting Weights:      {bundle.model.voting_weights}")


def version_callback(value: bool) -> None:
    if value:
        typer.echo("dmc-fraud version: 1.0.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", help="Show version and exit.", callback=version_callback, is_eager=True
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress visual banner and table outputs in scripted environments."
    ),
) -> None:
    """Retail Fraud Detection CLI (Data Mining Cup 2019)."""
    state["quiet"] = quiet


@app.command("train")
def train(
    train_data: Path = typer.Option(..., "--train-data", "-t", help="Path to training CSV (e.g. train.csv)"),
    output_bundle: Path = typer.Option(..., "--output-bundle", "-o", help="Destination path for .joblib model bundle"),
    folds: int = typer.Option(5, "--folds", "-k", help="Number of folds for OOF calibration"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to configuration YAML or JSON"),
    delimiter: str = typer.Option("|", "--delimiter", "-d", help="CSV field delimiter"),
    skip_calibration: bool = typer.Option(
        False, "--skip-calibration", help="Skip OOF calibration and use default thresholds (0.85, 0.75)"
    ),
) -> None:
    """Train the Hydra ensemble model, calibrate decision thresholds via OOF cross-validation,

    and package everything into a unified model artifact bundle (.joblib).
    """
    try:
        cfg: DMCConfig = load_config(config_path)
        active_delimiter = delimiter if delimiter != "|" else cfg.data_delimiter
        _echo(f"==> Loading training data from: {train_data}")
        df = load_data(train_data, delimiter=active_delimiter, validate=True, require_target=True)

        y_train = df[TARGET_COLUMN].values
        X_train_raw = df.drop(columns=[TARGET_COLUMN])

        fe = FraudFeatureEngineer(epsilon=cfg.features.epsilon)
        _echo("==> Fitting feature engineer and transforming features...")
        X_train_adv = fe.fit_transform(X_train_raw)

        if not skip_calibration:
            _echo(f"==> Starting {folds}-Fold Stratified Out-of-Fold Dual-Gate Calibration...")
            calibrator = DualGateCalibrator(
                config=cfg,
                cv_folds=folds,
                random_state=cfg.calibration.random_state,
            )
            calib_res = calibrator.calibrate(X_train_raw, y_train)
            thresholds = {1: calib_res.best_t1, 2: calib_res.best_t2}
            _echo(
                f"==> Calibration Succeeded! Best OOF Profit: € {calib_res.best_profit:,.2f} "
                f"(T1={calib_res.best_t1:.2f}, T2={calib_res.best_t2:.2f})"
            )
            oof_profit = calib_res.best_profit
        else:
            _echo(f"==> Skipping calibration. Using default thresholds {DEFAULT_CALIBRATED_THRESHOLDS}")
            thresholds = dict(DEFAULT_CALIBRATED_THRESHOLDS)
            oof_profit = 0.0

        _echo("==> Training final Hydra Soft-Voting Ensemble on full training set...")
        model = HydraClassifier(
            xgb_params=cfg.models.xgb_params,
            rf_params=cfg.models.rf_params,
            lr_params=cfg.models.lr_params,
            voting_weights=cfg.models.voting_weights,
            danger_trust_levels=cfg.models.danger_trust_levels,
            thresholds=thresholds,
        )
        model.fit(X_train_adv, y_train)

        metadata = {
            "version": cfg.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": len(df),
            "oof_profit": oof_profit,
            "thresholds": thresholds,
            "calibrated": not skip_calibration,
        }

        bundle = ModelBundle(
            model=model,
            feature_engineer=fe,
            thresholds=thresholds,
            metadata=metadata,
            config=cfg,
        )

        saved_path = bundle.save(output_bundle)
        _echo(f"==> Model bundle saved successfully to: {saved_path}")
        _echo(f"==> Checksum saved to: {saved_path.with_suffix(saved_path.suffix + '.sha256')}")

    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("calibrate")
def calibrate(
    train_data: Path = typer.Option(..., "--train-data", "-t", help="Path to training CSV"),
    folds: int = typer.Option(5, "--folds", "-k", help="Number of folds for OOF calibration"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to configuration"),
    delimiter: str = typer.Option("|", "--delimiter", "-d", help="CSV field delimiter"),
    output_bundle: Path | None = typer.Option(
        None, "--output-bundle", "-o", help="Optional path to update or save model bundle"
    ),
) -> None:
    """Run Stratified Out-of-Fold Dual-Gate threshold calibration on training data."""
    try:
        cfg = load_config(config_path)
        active_delimiter = delimiter if delimiter != "|" else cfg.data_delimiter
        _echo(f"==> Loading training data from: {train_data}")
        df = load_data(train_data, delimiter=active_delimiter, validate=True, require_target=True)

        y_train = df[TARGET_COLUMN].values
        X_train_raw = df.drop(columns=[TARGET_COLUMN])

        calibrator = DualGateCalibrator(config=cfg, cv_folds=folds)
        res = calibrator.calibrate(X_train_raw, y_train)

        _render_calibration_table(res)

        if output_bundle:
            if output_bundle.exists():
                bundle = ModelBundle.load(output_bundle)
                bundle.thresholds = {1: res.best_t1, 2: res.best_t2}
                bundle.metadata["last_calibrated"] = datetime.now(timezone.utc).isoformat()
                bundle.metadata["oof_profit"] = res.best_profit
                bundle.save(output_bundle)
                _echo(f"==> Updated thresholds in existing bundle: {output_bundle}")
            else:
                _echo(
                    f"Notice: Output bundle '{output_bundle}' does not exist to update. "
                    "Use 'dmc-fraud train' to create a new trained bundle with calibrated thresholds.",
                    color=typer.colors.YELLOW,
                )

    except Exception as exc:
        logger.error("Calibration failed: %s", exc, exc_info=True)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("predict")
def predict(
    model_bundle: Path = typer.Option(..., "--model-bundle", "-m", help="Path to .joblib model bundle"),
    input_data: Path = typer.Option(..., "--input-data", "-i", help="Path to input transactions CSV"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to output predictions CSV"),
    include_probabilities: bool = typer.Option(
        False, "--include-probabilities", "-p", help="Include estimated fraud probabilities in output"
    ),
    delimiter: str = typer.Option("|", "--delimiter", "-d", help="CSV field delimiter"),
    chunksize: int | None = typer.Option(
        None, "--chunksize", help="Optional batch chunksize for streaming large CSV transaction dumps"
    ),
) -> None:
    """Score transactions using a calibrated model artifact bundle."""
    try:
        _echo(f"==> Loading model bundle from: {model_bundle}")
        bundle = ModelBundle.load(model_bundle, verify_checksum=True)

        _echo(f"==> Loading input data from: {input_data}")
        output.parent.mkdir(parents=True, exist_ok=True)

        if chunksize and chunksize > 0:
            from dmc_fraud.data.loader import load_data_chunks

            _echo(f"==> Streaming predictions in chunks of {chunksize}...")
            total_rows = 0
            flagged_frauds = 0
            first_chunk = True

            for chunk_df in load_data_chunks(input_data, chunksize=chunksize, delimiter=delimiter, validate=True):
                if include_probabilities:
                    preds, probs = bundle.predict_with_proba(chunk_df, validate=False)
                    out_chunk = pd.DataFrame({"fraud": preds, "probability": probs})
                else:
                    preds = bundle.predict(chunk_df, validate=False)
                    out_chunk = pd.DataFrame({"fraud": preds})

                out_chunk.to_csv(output, mode="w" if first_chunk else "a", header=first_chunk, index=False)
                total_rows += len(preds)
                flagged_frauds += int(np.sum(preds == 1))
                first_chunk = False

            pct = (flagged_frauds / total_rows) if total_rows > 0 else 0.0
            _echo(f"==> Streamed {total_rows} predictions to: {output}")
            _echo(f"==> Flagged Frauds: {flagged_frauds} ({pct:.2%})")

        else:
            df = load_data(input_data, delimiter=delimiter, validate=True, require_target=False)

            if include_probabilities:
                preds, probs = bundle.predict_with_proba(df, validate=False)
                out_df = pd.DataFrame({"fraud": preds, "probability": probs})
            else:
                preds = bundle.predict(df, validate=False)
                out_df = pd.DataFrame({"fraud": preds})

            out_df.to_csv(output, index=False)
            _echo(f"==> Wrote {len(preds)} predictions to: {output}")
            _echo(f"==> Flagged Frauds: {int(np.sum(preds == 1))} ({np.mean(preds):.2%})")

    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("evaluate")
def evaluate(
    model_bundle: Path = typer.Option(..., "--model-bundle", "-m", help="Path to .joblib model bundle"),
    test_data: Path = typer.Option(..., "--test-data", "-t", help="Path to test transactions CSV"),
    labels: Path | None = typer.Option(
        None, "--labels", "-l", help="Optional path to ground truth CSV (with 'fraud' column)"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional path to save metrics JSON"),
    delimiter: str = typer.Option("|", "--delimiter", "-d", help="CSV field delimiter"),
) -> None:
    """Evaluate calibrated model performance on unseen holdout test data against DMC 2019 profit."""
    try:
        bundle = ModelBundle.load(model_bundle, verify_checksum=True)
        df_test = load_data(test_data, delimiter=delimiter, validate=True, require_target=False)

        if labels:
            df_labels = load_data(labels, delimiter=delimiter, validate=False)
            if TARGET_COLUMN not in df_labels.columns:
                raise ValueError(f"Ground truth file {labels} does not contain '{TARGET_COLUMN}' column.")
            y_test = df_labels[TARGET_COLUMN].values
        elif TARGET_COLUMN in df_test.columns:
            y_test = df_test[TARGET_COLUMN].values
        else:
            raise ValueError(
                "Ground truth labels must be provided either via --labels or as 'fraud' column in --test-data."
            )

        metrics = bundle.evaluate(df_test, y_test, validate=False)
        costs = bundle.config.costs

        _render_evaluation_table(metrics, costs)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            _echo(f"==> Saved metrics report to: {output}")

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("info")
def info(
    model_bundle: Path = typer.Option(..., "--model-bundle", "-m", help="Path to .joblib model bundle"),
) -> None:
    """Inspect model bundle metadata, calibrated thresholds, and constituent algorithms."""
    try:
        bundle = ModelBundle.load(model_bundle, verify_checksum=True)
        _render_bundle_info_table(bundle)

    except Exception as exc:
        logger.error("Info inspection failed: %s", exc, exc_info=True)
        typer.secho(f"ERROR: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
