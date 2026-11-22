# Operational Runbook — DMC 2019 Fraud Hydra Strategy

## 1. Service Overview
- **Artifact**: `dmc-fraud` CLI and `dmc_fraud` Python package.
- **Runtime**: Python 3.10–3.14 / Hardened Docker container (`appuser`, UID 10001).
- **Critical Guarantees**:
  - Out-of-Fold Dual-Gate threshold calibration prevents data leakage.
  - SHA-256 cryptographic checksum validation guards against tampered `.joblib` model artifacts.

## 2. Standard Operating Procedures

### Training & Artifact Bundle Creation
```bash
dmc-fraud train \
  --train-data data/train.csv \
  --output-bundle models/dmc_hydra.joblib \
  --folds 5
```

### Batch Scoring in Headless / Cron Pipelines
```bash
dmc-fraud --quiet predict \
  --model-bundle models/dmc_hydra.joblib \
  --input-data data/incoming.csv \
  --output predictions.csv \
  --chunksize 50000
```

### Docker Execution
```bash
docker compose run --rm dmc-fraud info --model-bundle /app/models/dmc_hydra.joblib
```

## 3. Incident Response & Troubleshooting

| Symptom | Probable Cause | Remediation Action |
|---|---|---|
| `ChecksumMismatchError` on `ModelBundle.load` | Corrupted transfer or unauthorized modification of `.joblib` | Quarantine artifact, re-fetch signed bundle from immutable artifact store, verify `.sha256` file. |
| `SchemaValidationError` on ingestion | Upstream CSV delimiter shift or negative monetary values | Inspect incoming CSV header with `head -n 5`, confirm delimiter (`\|`, `,`, `\t`), reject malformed batch. |
| Elevated False Positive Rate (Profit Drop) | Distribution drift in `trustLevel=1` or `trustLevel=2` | Re-run `dmc-fraud calibrate --train-data <recent_labeled.csv>` and compare economic profit against holdout baseline. |

## 4. Rollback Procedure
1. Pin previous verified model artifact bundle (`models/dmc_hydra_v<prev>.joblib` + `.sha256`).
2. Re-run `dmc-fraud evaluate` on the latest labeled window to verify positive economic profit.
3. Redeploy container image pinned to the prior git tag (`v1.0.0`).
