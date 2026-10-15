"""Unit tests for configuration schemas and file loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from dmc_fraud.config import DMCConfig, load_config


def test_default_config() -> None:
    """Verify default DMC 2019 economic costs and parameters."""
    cfg = DMCConfig()
    assert cfg.costs.profit_tp == 5.0
    assert cfg.costs.penalty_fp == 25.0
    assert cfg.costs.penalty_fn == 5.0
    assert cfg.costs.reward_tn == 0.0
    assert cfg.data_delimiter == "|"
    assert cfg.version == "1.0.0"


def test_yaml_config_roundtrip(tmp_path: Path) -> None:
    """Verify YAML configuration export and loading."""
    cfg = DMCConfig()
    cfg.costs.penalty_fp = 30.0
    yaml_file = tmp_path / "custom_config.yaml"

    cfg.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded_cfg = DMCConfig.from_yaml(yaml_file)
    assert loaded_cfg.costs.penalty_fp == 30.0
    assert loaded_cfg.costs.profit_tp == 5.0


def test_json_config_roundtrip(tmp_path: Path) -> None:
    """Verify JSON configuration export and loading."""
    cfg = DMCConfig()
    cfg.calibration.cv_folds = 10
    json_file = tmp_path / "custom_config.json"

    json_file.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    loaded_cfg = DMCConfig.from_json(json_file)
    assert loaded_cfg.calibration.cv_folds == 10


def test_load_config_helper(tmp_path: Path) -> None:
    """Verify load_config helper for None, YAML, and error paths."""
    # 1. Default config
    assert load_config(None).version == "1.0.0"

    # 2. YAML file
    cfg = DMCConfig()
    yaml_file = tmp_path / "conf.yml"
    cfg.to_yaml(yaml_file)
    assert load_config(yaml_file).data_delimiter == "|"

    # 3. Nonexistent file
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("missing_conf.yaml")

    # 4. Unsupported extension
    txt_file = tmp_path / "conf.txt"
    txt_file.write_text("costs: {}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_config(txt_file)


def test_feature_config_and_thresholds() -> None:
    """Verify DEFAULT_CALIBRATED_THRESHOLDS and FeatureConfig defaults."""
    from dmc_fraud.config import DEFAULT_CALIBRATED_THRESHOLDS, FeatureConfig

    assert DEFAULT_CALIBRATED_THRESHOLDS[1] == 0.85
    assert DEFAULT_CALIBRATED_THRESHOLDS[2] == 0.75

    fc = FeatureConfig()
    assert fc.epsilon == 1e-6

    cfg = DMCConfig()
    assert cfg.features.epsilon == 1e-6
