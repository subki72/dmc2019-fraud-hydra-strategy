"""Data loading utilities with delimiter detection and automatic schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dmc_fraud.data.schema import validate_dataframe

logger = logging.getLogger("dmc_fraud.data.loader")


class DataLoadError(Exception):
    """Raised when loading transaction data fails."""


def _detect_delimiter(file_path: str | Path, candidates: list[str] | None = None) -> str | None:
    """Detect delimiter by probing the first non-empty, non-comment line of the file.

    Skips blank lines and comment lines starting with '#'.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the CSV file.
    candidates : list[str], optional
        List of candidate delimiters to evaluate. Defaults to ['|', ',', ';', '\t'].

    Returns
    -------
    Optional[str]
        The best matching delimiter if found, otherwise None.
    """
    if candidates is None:
        candidates = ["|", ",", ";", "\t"]

    path = Path(file_path)
    try:
        first_valid_line: str = ""
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                stripped = line.lstrip("\ufeff").strip()
                if not stripped or stripped.startswith("#"):
                    continue
                first_valid_line = line
                break

        if not first_valid_line:
            return None

        best_delim: str | None = None
        max_count: int = 0
        for delim in candidates:
            cnt = first_valid_line.count(delim)
            if cnt > max_count:
                max_count = cnt
                best_delim = delim

        return best_delim if max_count > 0 else None
    except Exception:
        return None


def load_data(
    file_path: str | Path,
    delimiter: str = "|",
    validate: bool = True,
    require_target: bool = False,
    strict_bounds: bool = True,
) -> pd.DataFrame:
    """Load a tabular dataset from disk with delimiter verification and optional schema validation.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the dataset file (e.g. 'train.csv', 'test.csv').
    delimiter : str, default='|'
        Delimiter used in the CSV file. The DMC 2019 dataset uses pipe '|'.
    validate : bool, default=True
        Whether to run schema validation on the loaded DataFrame.
    require_target : bool, default=False
        Whether the 'fraud' target column is required.
    strict_bounds : bool, default=True
        Whether to enforce physical boundaries during validation.

    Returns
    -------
    pd.DataFrame
        Loaded and validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DataLoadError
        If parsing fails or file is empty.
    SchemaValidationError
        If schema validation fails.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    # Check file size (>500MB warning)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:
        logger.warning(
            "File size is %.1f MB (> 500MB). Consider using load_data_chunks() for memory-efficient streaming.",
            file_size_mb,
        )

    logger.info("Loading dataset from: %s (delimiter: '%s')", path, delimiter)

    try:
        df = pd.read_csv(path, sep=delimiter)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV from {path}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Dataset at {path} is empty (0 rows).")

    # Guard against incorrect delimiter detection (e.g. single column when another delimiter was used)
    if len(df.columns) == 1:
        logger.warning(
            "Only 1 column parsed with delimiter '%s'. Attempting delimiter autodetection for %s",
            delimiter,
            path,
        )
        detected = _detect_delimiter(path, candidates=[c for c in ["|", ",", ";", "\t"] if c != delimiter])
        candidates = [c for c in ["|", ",", ";", "\t"] if c != delimiter]
        if detected and detected in candidates:
            candidates.remove(detected)
            candidates.insert(0, detected)

        for fallback in candidates:
            try:
                test_df = pd.read_csv(path, sep=fallback, nrows=5)
                if len(test_df.columns) >= 9:
                    logger.info("Autodetected fallback delimiter '%s'", fallback)
                    df = pd.read_csv(path, sep=fallback)
                    break
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
                continue

    logger.info("Loaded dataset with shape %s from %s", df.shape, path)

    if validate:
        df = validate_dataframe(df, require_target=require_target, strict_bounds=strict_bounds)

    return df


def load_data_chunks(
    file_path: str | Path,
    chunksize: int = 50000,
    delimiter: str = "|",
    validate: bool = True,
    require_target: bool = False,
    strict_bounds: bool = True,
    on_bad_lines: str = "error",
):
    """Stream a large tabular dataset from disk in chunks with schema validation.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the dataset file.
    chunksize : int, default=50000
        Number of rows per chunk.
    delimiter : str, default='|'
        Delimiter used in the CSV file.
    validate : bool, default=True
        Whether to validate each chunk against the DMC schema.
    require_target : bool, default=False
        Whether the 'fraud' target column is required.
    strict_bounds : bool, default=True
        Whether to enforce bounds during validation.
    on_bad_lines : str, default='error'
        Specifies handling of bad lines: 'error', 'warn', or 'skip'.

    Yields
    ------
    pd.DataFrame
        Validated chunk DataFrame.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    logger.info("Streaming dataset in chunks of %d from: %s", chunksize, path)
    try:
        reader = pd.read_csv(path, sep=delimiter, chunksize=chunksize, on_bad_lines=on_bad_lines)
        for chunk in reader:
            if chunk.empty:
                continue
            if validate:
                chunk = validate_dataframe(chunk, require_target=require_target, strict_bounds=strict_bounds)
            yield chunk
    except Exception as exc:
        raise DataLoadError(f"Failed to stream CSV from {path}: {exc}") from exc
