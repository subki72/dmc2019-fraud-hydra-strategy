"""Pytest fixtures and synthetic DMC 2019 transaction data generators."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmc_fraud.data.schema import TARGET_COLUMN


def generate_synthetic_dmc_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic DMC 2019 transaction data with exact schema.

    Enforces that the majority of frauds occur in trustLevel 1 and 2 to match
    the real Data Mining Cup 2019 distribution.
    """
    rng = np.random.RandomState(random_state)

    trust_levels = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.25, 0.25, 0.15, 0.15, 0.1, 0.1])
    scan_time = rng.uniform(20.0, 1800.0, size=n_samples)
    grand_total = rng.uniform(5.0, 500.0, size=n_samples)
    line_item_voids = rng.poisson(lam=1.5, size=n_samples)
    scans_no_reg = rng.poisson(lam=1.0, size=n_samples)
    qty_mods = rng.poisson(lam=0.8, size=n_samples)

    # Scanned items per second (realistic self-checkout speed)
    total_items = np.maximum(1, rng.poisson(lam=15, size=n_samples))
    scanned_items_per_sec = total_items / scan_time
    value_per_sec = grand_total / scan_time
    voids_per_pos = line_item_voids / total_items

    # Generate fraud labels: high-risk in trustLevel 1 & 2
    fraud = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        lvl = trust_levels[i]
        if lvl == 1:
            # 20% fraud probability for trustLevel 1
            fraud[i] = 1 if rng.rand() < 0.20 else 0
        elif lvl == 2:
            # 10% fraud probability for trustLevel 2
            fraud[i] = 1 if rng.rand() < 0.10 else 0
        else:
            # 0% fraud probability for high-trust customers
            fraud[i] = 0

    df = pd.DataFrame(
        {
            "trustLevel": trust_levels,
            "totalScanTimeInSeconds": scan_time,
            "grandTotal": grand_total,
            "lineItemVoids": line_item_voids,
            "scansWithoutRegistration": scans_no_reg,
            "quantityModifications": qty_mods,
            "scannedLineItemsPerSecond": scanned_items_per_sec,
            "valuePerSecond": value_per_sec,
            "lineItemVoidsPerPosition": voids_per_pos,
            TARGET_COLUMN: fraud,
        }
    )

    return df


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Fixture providing a synthetic training DataFrame with 150 rows."""
    return generate_synthetic_dmc_data(n_samples=150, random_state=42)


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Fixture providing a synthetic test DataFrame without target column."""
    df = generate_synthetic_dmc_data(n_samples=60, random_state=99)
    return df.drop(columns=[TARGET_COLUMN])


@pytest.fixture
def synthetic_test_with_target() -> pd.DataFrame:
    """Fixture providing a synthetic test DataFrame with ground truth."""
    return generate_synthetic_dmc_data(n_samples=60, random_state=99)


@pytest.fixture
def temp_data_dir(
    tmp_path: Path,
    synthetic_train_df: pd.DataFrame,
    synthetic_test_df: pd.DataFrame,
    synthetic_test_with_target: pd.DataFrame,
) -> Generator[Path, None, None]:
    """Fixture creating temporary pipe-separated CSV files on disk."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    labels_path = tmp_path / "realclass.csv"

    synthetic_train_df.to_csv(train_path, sep="|", index=False)
    synthetic_test_df.to_csv(test_path, sep="|", index=False)
    synthetic_test_with_target[[TARGET_COLUMN]].to_csv(labels_path, sep="|", index=False)

    yield tmp_path
