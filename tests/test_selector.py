"""Tests for monthly portfolio selection."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.selector import select_monthly_portfolio, select_portfolio


def test_sector_concentration_limit() -> None:
    """No sector should exceed the configured maximum count."""
    universe, factors = _frames(40)
    universe.loc[:11, "sector"] = "Semiconductors"
    universe["universe_role"] = "Growth"
    universe.loc[12:16, "universe_role"] = "Core"
    universe.loc[17:19, "universe_role"] = "Defensive"

    result = select_monthly_portfolio(factors, universe, "2024-06-30")

    assert len(result.selected_portfolio) == 30
    assert result.selected_portfolio["sector"].value_counts().max() <= 6
    assert "sector_limit" in result.excluded_stocks["exclusion_reason"].tolist()


def test_kosdaq_count_limit() -> None:
    """KOSDAQ names should be capped at ten even when they score highest."""
    universe, factors = _frames(40)
    universe.loc[:19, "market"] = "KOSDAQ"
    universe.loc[20:, "market"] = "KOSPI"

    result = select_monthly_portfolio(factors, universe, "2024-06-30")

    assert len(result.selected_portfolio) == 30
    assert int((result.selected_portfolio["market"] == "KOSDAQ").sum()) <= 10
    assert "kosdaq_limit" in result.excluded_stocks["exclusion_reason"].tolist()


def test_new_listing_is_excluded() -> None:
    """Stocks with fewer than 252 trading days by rebalance date should be excluded."""
    universe, factors = _frames(35)
    universe.loc[0, "listing_date"] = pd.Timestamp("2024-02-01")

    result = select_monthly_portfolio(factors, universe, "2024-06-30")

    assert "000001" not in result.selected_portfolio["ticker"].tolist()
    row = result.excluded_stocks.loc[result.excluded_stocks["ticker"] == "000001"].iloc[0]
    assert row["exclusion_reason"] == "insufficient_listing_history"


def test_minimum_core_and_defensive_roles_are_met() -> None:
    """Role minimums should reserve enough room for Core and Defensive candidates."""
    universe, factors = _frames(40)
    universe["universe_role"] = "Growth"
    universe.loc[32:36, "universe_role"] = "Core"
    universe.loc[37:39, "universe_role"] = "Defensive"

    result = select_monthly_portfolio(factors, universe, "2024-06-30")
    role_counts = result.selected_portfolio["universe_role"].value_counts()

    assert len(result.selected_portfolio) == 30
    assert int(role_counts["Core"]) >= 5
    assert int(role_counts["Defensive"]) >= 3
    assert "reserved_for_min_role" in result.excluded_stocks["exclusion_reason"].tolist()


def test_underfilled_result_logs_shortage_reasons(caplog: pytest.LogCaptureFixture) -> None:
    """Underfilled selections should return exclusions and log likely causes."""
    universe, factors = _frames(12)
    constraints = PortfolioConstraints(target_size=30, relaxation_order=())

    with caplog.at_level(logging.WARNING, logger="src.portfolio.selector"):
        selected, excluded, summary = select_portfolio(
            factors,
            universe,
            "2024-06-30",
            constraints=constraints,
        )

    assert len(selected) == 12
    assert len(excluded) == 0
    assert summary["shortage"] == 18
    assert summary["shortage_reasons"] == ["eligible universe smaller than target_size"]
    assert "underfilled" in caplog.text


def _frames(count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = [f"{idx:06d}" for idx in range(1, count + 1)]
    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "market": ["KOSPI"] * count,
            "sector": [f"Sector{idx % 8}" for idx in range(count)],
            "universe_role": [
                "Core" if idx < 6 else "Defensive" if idx < 9 else "Growth" for idx in range(count)
            ],
            "listing_date": [pd.Timestamp("2010-01-01")] * count,
            "data_start_date": [pd.Timestamp("2010-01-01")] * count,
            "is_active": [True] * count,
        }
    )
    factors = pd.DataFrame(
        {
            "calculation_date": [pd.Timestamp("2024-06-30")] * count,
            "available_date": [pd.Timestamp("2024-06-29")] * count,
            "ticker": tickers,
            "composite_score": [float(count - idx) for idx in range(count)],
            "momentum_score": [1.0] * count,
            "relative_strength_score": [1.0] * count,
            "quality_score": [1.0] * count,
            "growth_score": [1.0] * count,
            "low_volatility_score": [pd.NA] * count,
            "liquidity_score": [pd.NA] * count,
            "avg_trading_value_20d": [1_000_000_000.0] * count,
            "zero_volume_ratio_60d": [0.0] * count,
        }
    )
    return universe, factors
