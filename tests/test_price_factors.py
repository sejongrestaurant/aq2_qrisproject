"""Tests for price-based factor calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.common import cross_sectional_zscore, winsorize_series
from src.factors.liquidity import calculate_liquidity
from src.factors.low_volatility import calculate_low_volatility
from src.factors.momentum import calculate_momentum
from src.factors.relative_strength import calculate_relative_strength


def test_momentum_uses_only_past_data() -> None:
    """Changing future prices should not alter earlier calculation dates."""
    prices = _make_price_frame(periods=420)
    baseline = calculate_momentum(prices)

    modified = prices.copy()
    future_mask = modified["date"] > pd.Timestamp("2024-06-28")
    modified.loc[future_mask & (modified["ticker"] == "005930"), "adjusted_close"] *= 100
    changed = calculate_momentum(modified)

    baseline_value = _factor_value(
        baseline,
        ticker="005930",
        calculation_date="2024-06-28",
        column="momentum_raw",
    )
    changed_value = _factor_value(
        changed,
        ticker="005930",
        calculation_date="2024-06-28",
        column="momentum_raw",
    )

    assert baseline_value == changed_value


def test_momentum_returns_nan_when_observations_are_insufficient() -> None:
    """Insufficient lookback windows should return NaN rather than fabricated values."""
    prices = _make_price_frame(periods=80)

    result = calculate_momentum(prices)

    assert result["momentum_raw"].isna().all()


def test_relative_strength_combines_market_and_sector_excess_returns() -> None:
    """Relative strength should include both market and sector excess returns."""
    prices = _make_price_frame(periods=180)
    calculation_date = prices["date"].max()
    market_returns = pd.DataFrame(
        [
            {"calculation_date": calculation_date, "market": "KOSPI", "market_return_6m": 0.01},
            {"calculation_date": calculation_date, "market": "KOSDAQ", "market_return_6m": 0.02},
        ]
    )
    sector_map = prices[["ticker", "sector"]].drop_duplicates()

    result = calculate_relative_strength(prices, market_returns, sector_map=sector_map)
    latest = result.loc[result["calculation_date"] == calculation_date]

    assert {"market_excess_return", "sector_excess_return", "relative_strength_score"} <= set(
        latest.columns
    )
    assert latest["relative_strength_raw"].notna().any()


def test_low_volatility_gives_higher_score_to_lower_risk_stock() -> None:
    """Lower realized risk should map to a higher low-volatility score."""
    prices = _make_price_frame(periods=320)

    result = calculate_low_volatility(prices)
    latest_date = result["calculation_date"].max()
    latest = result.loc[result["calculation_date"] == latest_date]
    stable_score = latest.loc[latest["ticker"] == "005930", "low_volatility_score"].iloc[0]
    volatile_score = latest.loc[latest["ticker"] == "000660", "low_volatility_score"].iloc[0]

    assert stable_score > volatile_score


def test_liquidity_marks_low_liquidity_ineligible() -> None:
    """Liquidity factor should expose eligibility for illiquid securities."""
    prices = _make_price_frame(periods=90)
    low_liquidity_mask = prices["ticker"] == "000660"
    prices.loc[low_liquidity_mask, "trading_value"] = 10_000.0
    prices.loc[low_liquidity_mask, "volume"] = 0.0

    result = calculate_liquidity(prices, min_median_trading_value=100_000.0)
    latest = result.loc[result["calculation_date"] == result["calculation_date"].max()]

    assert bool(latest.loc[latest["ticker"] == "005930", "is_liquidity_eligible"].iloc[0]) is True
    assert bool(latest.loc[latest["ticker"] == "000660", "is_liquidity_eligible"].iloc[0]) is False


def test_common_zscore_handles_zero_std_and_winsorization() -> None:
    """Common helpers should handle constant samples and outliers."""
    constant_zscore = cross_sectional_zscore(pd.Series([1.0, 1.0, 1.0]))
    winsorized = winsorize_series(pd.Series([1.0, 2.0, 100.0]), 0.0, 0.5)

    assert constant_zscore.tolist() == [0.0, 0.0, 0.0]
    assert winsorized.max() == 2.0


def _make_price_frame(periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    tickers = [
        ("005930", "KOSPI", "Technology", 0.0005, 0.002),
        ("000660", "KOSPI", "Technology", 0.0008, 0.03),
        ("035720", "KOSDAQ", "Internet", 0.0002, 0.01),
    ]
    for ticker, market, sector, drift, noise_scale in tickers:
        for index, current_date in enumerate(dates):
            seasonal_noise = np.sin(index / 5) * noise_scale
            close = 100.0 * (1.0 + drift) ** index * (1.0 + seasonal_noise)
            volume = 1000.0 + index
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "market": market,
                    "sector": sector,
                    "adjusted_close": close,
                    "volume": volume,
                    "trading_value": close * volume,
                }
            )
    return pd.DataFrame(rows)


def _factor_value(
    df: pd.DataFrame,
    *,
    ticker: str,
    calculation_date: str,
    column: str,
) -> float:
    matched = df.loc[
        (df["ticker"] == ticker) & (df["calculation_date"] == pd.Timestamp(calculation_date)),
        column,
    ]
    assert not matched.empty
    return float(matched.iloc[0])
