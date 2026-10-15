# Retail Fraud Detection (Data Mining Cup 2019) — Hydra Strategy

[![CI Pipeline](https://github.com/subki72/dmc2019-fraud-hydra-strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/dmc2019-fraud-hydra-strategy/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-85%20passed-brightgreen.svg)](tests/)
[![Test Coverage](https://img.shields.io/badge/coverage-92.18%25-brightgreen.svg)](pyproject.toml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checking: Mypy](https://img.shields.io/badge/type%20checker-mypy-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-grade modular machine learning library and multi-command CLI (`dmc-fraud`) designed to maximize financial profit on retail self-checkout scanning data under the **Data Mining Cup (DMC) 2019** asymmetric cost objective.

---

## The Business Problem & Asymmetric Cost Matrix

In retail self-checkout systems, fraud detection is inherently cost-sensitive and class-imbalanced. A standard model optimizing for raw Accuracy or ROC-AUC will lead to economic loss due to customer friction caused by false accusations.

Under the official DMC 2019 evaluation matrix:
- **True Positive (Caught Fraud):** `+€5.00` (Recovered stolen merchandise)
- **False Negative (Missed Fraud):** `-€5.00` (Unchecked retail shrinkage)
- **False Positive (False Accusation):** `-€25.00` (Customer friction, manual search delay, lost goodwill)
- **True Negative (Clean Customer):** `€0.00` (Frictionless checkout)

$$\text{Economic Profit} = (\text{TP} \times 5) - (\text{FP} \times 25) - (\text{FN} \times 5)$$

Because a False Positive is **5x more expensive** than a Missed Fraud, models must maintain surgical precision in low-trust customer tiers while avoiding accusations against trusted shoppers.

---

## Architectural Solution: Hydra Ensemble & Dual-Gate Strategy

The `dmc-fraud` architecture consists of four tightly engineered layers:

```
[Raw Tabular CSV (sep='|' or auto-detected)]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│        Data Ingestion & Schema Validation              │
│  - Multi-delimiter auto-detection (pipe, comma, tab)   │
│  - Strict schema boundary checks (Pydantic / Pandera)  │
│  - Streaming batch chunk loader for large files        │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│     FraudFeatureEngineer (Scikit-Learn Transformer)    │
│  - Reconstruction: scannedLineItemsTotal               │
│  - Safe non-zero division ratios: avgTimePerItem,      │
│    voidsPerTotalItem, noRegPerItem, pricePerItem       │
│  - Non-linear transforms & behavioral interactions     │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│              The Hydra Soft-Voting Ensemble            │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐│
│  │ XGBoost (2.0) │  │ RandForest(1.0)│  │ LogReg (0.5) ││
│  └───────────────┘  └───────────────┘  └──────────────┘│
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│   Out-of-Fold (OOF) Dual-Gate Risk Calibrator          │
│  - Stratified 5-Fold on training set (ZERO leakage!)   │
│  - Independent disjoint grid search: O(2K) complexity  │
│  - Conservative tie-breaking (favors higher threshold) │
│  - Trust Level 1 (Strict: ~0.85)                       │
│  - Trust Level 2 (Moderate: ~0.75)                     │
│  - Trust Level 3-6 (Pass through: 0 fraud flags)       │
└────────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│   Unified Model Artifact Bundle (.joblib + SHA256)     │
│  - Cryptographic checksum verification on load         │
│  - Path confinement protection against directory jumps │
│  - Single-pass joint inference (predict_with_proba)    │
└────────────────────────────────────────────────────────┘
```

1. **Systematic Feature Engineering (`FraudFeatureEngineer`):**
   Reconstructs the true scan volume (`scannedLineItemsTotal`), derives ratio-based behavioral metrics (`avgTimePerItem`, `voidsPerTotalItem`, `noRegPerItem`, `pricePerItem`) using safe conditional division (`np.where(total_items == 0, 0.0, ...)`), and interaction features (`interaction_trust_time`, `interaction_void_price`).
2. **The Hydra Ensemble (`HydraClassifier`):**
   Combines three diverse learning algorithms via Soft-Voting:
   - **XGBoost:** Gradient boosting for high-order non-linear patterns (scale_pos_weight tuned to imbalance).
   - **Random Forest:** Bagging to control variance and prevent decision tree overfitting.
   - **Logistic Regression (StandardScaler):** Regularized linear baseline ensuring well-behaved probability margins.
   - Weighting: `2.0 : 1.0 : 0.5`.
3. **Leakage-Free Dual-Gate Thresholding (`DualGateCalibrator`):**
   Calibrates decision boundaries strictly via **Stratified K-Fold Out-of-Fold (OOF)** predictions on the training data. The holdout test set is never seen or snooped during calibration, ensuring verified out-of-sample generalization.
4. **Cryptographic Model Packaging (`ModelBundle`):**
   Packages trained weights, calibrated thresholds, preprocessors, and metadata into a unified `.joblib` archive with companion `.sha256` checksum verification and optional `base_dir` path confinement.

---

## Installation

### From Source using uv (Recommended)
```bash
git clone https://github.com/subki72/dmc2019-fraud-hydra-strategy.git
cd dmc2019-fraud-hydra-strategy
uv pip install -e .
```

### From Source using standard pip
```bash
pip install -e .
```

### Developer & QA Installation
```bash
pip install -e ".[dev]"
```

---

## CLI Quickstart Guide

The `dmc-fraud` command-line tool provides multi-command workflows for training, calibrating, scoring, and evaluating transactions.

```bash
# Display CLI help and global options
dmc-fraud --help
```

### 1. Train & Calibrate Model Bundle
Trains the Hydra ensemble and derives optimal thresholds via Stratified 5-Fold OOF cross-validation:
```bash
dmc-fraud train \
  --train-data data/train.csv \
  --output-bundle models/dmc_hydra.joblib \
  --folds 5
```

### 2. Standalone Out-of-Fold Calibration
Runs OOF threshold calibration without re-training from scratch:
```bash
dmc-fraud calibrate \
  --train-data data/train.csv \
  --folds 5
```

### 3. High-Throughput Batch Scoring (`predict`)
Scores transactions using the calibrated bundle with optional streaming chunking:
```bash
dmc-fraud predict \
  --model-bundle models/dmc_hydra.joblib \
  --input-data data/test.csv \
  --output predictions.csv \
  --include-probabilities \
  --chunksize 50000
```

### 4. Holdout Profit Evaluation (`evaluate`)
Evaluates profit against ground-truth labels and displays a formatted confusion matrix:
```bash
dmc-fraud evaluate \
  --model-bundle models/dmc_hydra.joblib \
  --test-data data/test.csv \
  --labels data/DMC-2019-realclass.csv \
  --output evaluation_report.json
```

### 5. Inspect Model Artifact Bundle (`info`)
Inspects model metadata, creation timestamp, commit hash, and active thresholds:
```bash
dmc-fraud info --model-bundle models/dmc_hydra.joblib
```

### 6. Scripted / Cron Automation (`--quiet`)
Suppresses visual Rich banners for headless batch scripts and CI pipelines:
```bash
dmc-fraud --quiet predict \
  --model-bundle models/dmc_hydra.joblib \
  --input-data data/test.csv \
  --output predictions.csv
```

---

## Python API Usage

You can embed `dmc_fraud` directly into your Python backend services or ETL pipelines:

```python
import pandas as pd
from dmc_fraud import ModelBundle, calculate_dmc_profit, load_data

# 1. Load an artifact bundle with automatic SHA-256 integrity verification
bundle = ModelBundle.load(
    "models/dmc_hydra.joblib",
    verify_checksum=True,
    base_dir="models/"  # Path confinement protection
)

# 2. Ingest transaction batch with schema validation
transactions_df = load_data("data/new_batch.csv", delimiter="|", validate=True)

# 3. Predict binary fraud flags using calibrated dual-gate thresholds
fraud_predictions = bundle.predict(transactions_df)

# 4. Predict raw calibrated fraud probabilities (single-pass)
fraud_predictions, fraud_probabilities = bundle.predict_with_proba(transactions_df)

# 5. Calculate economic profit on labeled data
profit = calculate_dmc_profit(y_true=ground_truth, y_pred=fraud_predictions)
print(f"Generated Profit: € {profit:,.2f}")
```

---

## Input Data Schema

The dataset follows the pipe-separated (`sep='|'`) tabular structure specified in the Data Mining Cup 2019:

| Column Name | Type | Constraints | Description |
|---|:---:|:---:|---|
| `trustLevel` | `int` | `1..6` | Customer trust level (1 = lowest, 6 = highest verified trust) |
| `totalScanTimeInSeconds` | `float` | $\ge 0$ | Total duration of the checkout session in seconds |
| `grandTotal` | `float` | $\ge 0$ | Total monetary value of scanned items in Euros (€) |
| `lineItemVoids` | `int` | $\ge 0$ | Total count of voided item scans |
| `scansWithoutRegistration` | `int` | $\ge 0$ | Count of barcode scans without item registration |
| `quantityModifications` | `int` | $\ge 0$ | Count of manual quantity modifications |
| `scannedLineItemsPerSecond` | `float` | $\ge 0$ | Scanning rate (items / second) |
| `valuePerSecond` | `float` | $\ge 0$ | Monetary turnover rate (Euros / second) |
| `lineItemVoidsPerPosition` | `float` | $\ge 0$ | Voided items ratio per scanned position |
| `fraud` *(train only)* | `int` | `{0, 1}` | Ground truth label (1 = fraud, 0 = clean customer) |

---

## Verification & Quality Assurance

The codebase adheres to rigorous production software engineering standards:

```bash
# Run full automated Pytest test suite with coverage gate (>= 80%)
pytest

# Run Ruff linter and code style enforcement
ruff check src tests
ruff format --check src tests

# Run Mypy static type checking
mypy src tests
```

### Test Suite Summary
- **Total Unit & Integration Tests:** **85 passing tests** across 9 test suites (`test_bundle`, `test_calibrator`, `test_cli`, `test_config`, `test_features`, `test_hydra`, `test_loader`, `test_metrics`, `test_schema`).
- **Statement Coverage:** **92.18%** (exceeds the 80% CI requirement).
- **Static Typing:** Fully typed with zero Mypy errors under strict settings across 27 source files.
- **CI Matrix:** Automated GitHub Actions matrix across Python 3.10, 3.11, 3.12, 3.13 on Linux & Windows.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
