"""Universe loading and query functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.universe.validator import raise_for_validation_errors, validate_universe


def load_universe(path: Path) -> pd.DataFrame:
    """Load, normalize, and validate the Korean equity universe CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path, dtype={"ticker": "string"})
    normalized_df = _normalize_universe(df)
    raise_for_validation_errors(validate_universe(normalized_df))
    return normalized_df


def get_active_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose active flag is true."""
    if "is_active" not in df.columns:
        raise KeyError("Column is_active is required")
    return df.loc[df["is_active"]].copy()


def get_point_in_time_universe(df: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return securities whose data start date is on or before the given date."""
    if "data_start_date" not in df.columns:
        raise KeyError("Column data_start_date is required")

    as_of_timestamp = pd.Timestamp(as_of_date)
    if pd.isna(as_of_timestamp):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    data_start_date = pd.to_datetime(df["data_start_date"], errors="raise")
    return df.loc[data_start_date <= as_of_timestamp].copy()


def get_universe_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return high-level universe counts for research diagnostics."""
    required_columns = {"market", "sector", "universe_role", "data_start_date"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise KeyError(f"Missing columns for summary: {', '.join(missing_columns)}")

    data_start_date = pd.to_datetime(df["data_start_date"], errors="raise")
    return {
        "total_count": int(len(df)),
        "market_counts": _value_counts(df["market"]),
        "sector_counts": _value_counts(df["sector"]),
        "role_counts": _value_counts(df["universe_role"]),
        "listing_year_counts": {
            int(year): int(count)
            for year, count in data_start_date.dt.year.value_counts().sort_index().items()
        },
    }


def _normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV dtypes used by validation and downstream modules."""
    normalized_df = df.copy()

    if "ticker" in normalized_df.columns:
        normalized_df["ticker"] = normalized_df["ticker"].map(_normalize_ticker)

    if "data_start_date" in normalized_df.columns:
        normalized_df["data_start_date"] = pd.to_datetime(
            normalized_df["data_start_date"], errors="raise"
        )

    if "is_active" in normalized_df.columns:
        normalized_df["is_active"] = normalized_df["is_active"].map(_normalize_bool)

    return normalized_df


def _normalize_ticker(value: object) -> str:
    """Convert ticker values to zero-padded six-character strings."""
    if pd.isna(value):
        raise ValueError("ticker contains a missing value")

    ticker = str(value).strip()
    if ticker.endswith(".0"):
        ticker = ticker[:-2]
    if not ticker.isdigit():
        raise ValueError(f"ticker must contain only digits: {value}")
    if len(ticker) > 6:
        raise ValueError(f"ticker must be at most 6 digits: {value}")
    return ticker.zfill(6)


def _normalize_bool(value: object) -> bool:
    """Convert common CSV boolean values to bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError("is_active contains a missing value")

    normalized_value = str(value).strip().lower()
    if normalized_value in {"true", "1", "yes", "y"}:
        return True
    if normalized_value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"is_active must be a boolean-like value: {value}")


def _value_counts(series: pd.Series) -> dict[str, int]:
    """Return deterministic value counts with plain Python scalar values."""
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


__all__ = [
    "get_active_universe",
    "get_point_in_time_universe",
    "get_universe_summary",
    "load_universe",
]
