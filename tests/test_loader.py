"""Unit tests for dataset loading and delimiter autodetection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dmc_fraud.data.loader import DataLoadError, load_data


def test_load_data_pipe_delimited(temp_data_dir: Path) -> None:
    """Standard pipe-separated CSV should load cleanly."""
    train_file = temp_data_dir / "train.csv"
    df = load_data(train_file, delimiter="|", validate=True, require_target=True)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "trustLevel" in df.columns
    assert "fraud" in df.columns


def test_load_data_comma_fallback(tmp_path: Path, synthetic_train_df: pd.DataFrame) -> None:
    """If file is comma-delimited but loaded with delimiter='|', autodetect handles it."""
    comma_file = tmp_path / "comma_train.csv"
    synthetic_train_df.to_csv(comma_file, sep=",", index=False)

    df = load_data(comma_file, delimiter="|", validate=True, require_target=True)
    assert len(df.columns) == len(synthetic_train_df.columns)
    assert len(df) == len(synthetic_train_df)


def test_load_data_nonexistent_file() -> None:
    """Nonexistent file path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Dataset file does not exist"):
        load_data("nonexistent_path_to_file.csv")


def test_load_data_empty_file(tmp_path: Path) -> None:
    """Empty CSV file must raise DataLoadError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="Failed to read CSV|empty"):
        load_data(empty_file)


def test_load_data_pipe_fallback_when_comma_specified(tmp_path: Path, synthetic_train_df: pd.DataFrame) -> None:
    """If file is pipe-delimited but requested with delimiter=',', autodetection recovers pipe."""
    pipe_file = tmp_path / "pipe_train.csv"
    synthetic_train_df.to_csv(pipe_file, sep="|", index=False)

    df = load_data(pipe_file, delimiter=",", validate=True, require_target=True)
    assert len(df.columns) == len(synthetic_train_df.columns)
    assert len(df) == len(synthetic_train_df)


def test_load_data_chunks_streaming(tmp_path: Path, synthetic_train_df: pd.DataFrame) -> None:
    """Verify load_data_chunks yields valid chunks for streaming batch operations."""
    from dmc_fraud.data.loader import load_data_chunks

    data_file = tmp_path / "stream_data.csv"
    synthetic_train_df.to_csv(data_file, sep="|", index=False)

    chunks = list(load_data_chunks(data_file, chunksize=50, delimiter="|", validate=True))
    assert len(chunks) == 3  # 150 rows in chunks of 50
    total_streamed = sum(len(c) for c in chunks)
    assert total_streamed == len(synthetic_train_df)
    assert all("trustLevel" in c.columns for c in chunks)


def test_detect_delimiter_with_blank_lines_and_comments(tmp_path: Path) -> None:
    """_detect_delimiter must skip leading blank lines and '#' comment lines."""
    from dmc_fraud.data.loader import _detect_delimiter

    f = tmp_path / "commented.csv"
    f.write_text(
        "\n\n# Data Mining Cup 2019 dataset\n# Metadata header comment\n\ntrustLevel|totalScanTimeInSeconds|grandTotal\n1|100|50\n",
        encoding="utf-8",
    )

    delim = _detect_delimiter(f)
    assert delim == "|"


def test_detect_delimiter_empty_or_only_comments(tmp_path: Path) -> None:
    """_detect_delimiter returns None if file contains only comments and blank lines."""
    from dmc_fraud.data.loader import _detect_delimiter

    f = tmp_path / "only_comments.csv"
    f.write_text("# Line 1\n# Line 2\n\n\n", encoding="utf-8")

    assert _detect_delimiter(f) is None


def test_load_data_chunks_on_bad_lines(tmp_path: Path) -> None:
    """load_data_chunks with on_bad_lines='skip' successfully skips malformed rows."""
    from dmc_fraud.data.loader import load_data_chunks

    f = tmp_path / "bad_rows.csv"
    # Row 2 is malformed (too many fields)
    content = (
        "trustLevel|totalScanTimeInSeconds|grandTotal|lineItemVoids|scansWithoutRegistration|"
        "quantityModifications|scannedLineItemsPerSecond|valuePerSecond|lineItemVoidsPerPosition|fraud\n"
        "1|100.0|50.0|2|1|0|0.1|0.5|0.2|0\n"
        "1|100.0|50.0|2|1|0|0.1|0.5|0.2|EXTRA|FIELD|0\n"
        "2|200.0|70.0|1|0|0|0.2|0.7|0.1|1\n"
    )
    f.write_text(content, encoding="utf-8")

    chunks = list(load_data_chunks(f, chunksize=10, delimiter="|", validate=True, on_bad_lines="skip"))
    assert len(chunks) == 1
    assert len(chunks[0]) == 2


def test_load_data_large_file_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """load_data emits warning when file size > 500MB."""
    import logging
    import os

    train_file = tmp_path / "large.csv"
    train_file.write_text(
        "trustLevel|totalScanTimeInSeconds|grandTotal|lineItemVoids|scansWithoutRegistration|"
        "quantityModifications|scannedLineItemsPerSecond|valuePerSecond|lineItemVoidsPerPosition|fraud\n"
        "1|100.0|50.0|2|1|0|0.1|0.5|0.2|0\n",
        encoding="utf-8",
    )

    target_str = str(train_file.resolve()).lower()
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        res = real_stat(self, *args, **kwargs)
        if str(self).lower() == target_str:
            return os.stat_result(
                (
                    res.st_mode,
                    res.st_ino,
                    res.st_dev,
                    res.st_nlink,
                    res.st_uid,
                    res.st_gid,
                    600 * 1024 * 1024,
                    res.st_atime,
                    res.st_mtime,
                    res.st_ctime,
                )
            )
        return res

    monkeypatch.setattr(Path, "stat", fake_stat)

    with caplog.at_level(logging.WARNING):
        df = load_data(train_file, delimiter="|", validate=True)
        assert not df.empty
        assert any("Consider using load_data_chunks()" in r.message for r in caplog.records)


def test_detect_delimiter_with_utf8_bom_and_comments(tmp_path: Path) -> None:
    """_detect_delimiter must correctly skip leading comments even with UTF-8 BOM."""
    from dmc_fraud.data.loader import _detect_delimiter

    bom_file = tmp_path / "bom_commented.csv"
    bom_file.write_bytes(b"\xef\xbb\xbf# Leading comment 1\n# Leading comment 2\n\ntrustLevel|grandTotal\n1|50.0\n")

    detected = _detect_delimiter(bom_file)
    assert detected == "|"
