"""Tests for universe loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.universe.loader import get_point_in_time_universe, load_universe
from src.universe.validator import validate_universe

UNIVERSE_PATH = Path("data/universe/korea_active_etf_universe_100.csv")


def test_load_universe_preserves_leading_zero_ticker() -> None:
    """Numeric-looking tickers should be normalized to six-character strings."""
    df = load_universe(UNIVERSE_PATH)

    samsung = df.loc[df["company_name"] == "삼성전자"].iloc[0]

    assert samsung["ticker"] == "005930"


def test_validate_universe_detects_duplicate_ticker() -> None:
    """Duplicate ticker values should be reported by validation."""
    df = load_universe(UNIVERSE_PATH)
    df.loc[df.index[1], "ticker"] = df.loc[df.index[0], "ticker"]

    errors = validate_universe(df)

    assert any("Duplicate ticker" in error for error in errors)


def test_validate_universe_detects_invalid_market() -> None:
    """Only KOSPI and KOSDAQ should be accepted as market values."""
    df = load_universe(UNIVERSE_PATH)
    df.loc[df.index[0], "market"] = "NYSE"

    errors = validate_universe(df)

    assert any("Invalid market" in error for error in errors)


def test_point_in_time_universe_excludes_not_yet_listed_stock() -> None:
    """Rows with a later data_start_date should be excluded for earlier dates."""
    df = pd.DataFrame(
        {
            "ticker": ["005930", "403870"],
            "data_start_date": pd.to_datetime(["2010-01-01", "2022-07-15"]),
            "is_active": [True, True],
        }
    )

    point_in_time_df = get_point_in_time_universe(df, "2020-12-31")

    assert point_in_time_df["ticker"].tolist() == ["005930"]


def test_load_universe_contains_exactly_100_rows() -> None:
    """The project universe file should contain exactly 100 securities."""
    df = load_universe(UNIVERSE_PATH)

    assert len(df) == 100
