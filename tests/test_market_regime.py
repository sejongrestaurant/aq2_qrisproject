"""Tests for rule-based market regime classification."""

from __future__ import annotations

import pandas as pd

from src.config.regime_config import RegimeConfig
from src.regime.market_regime import (
    calculate_regime_statistics,
    classify_market_regime,
    get_equity_cash_allocation,
)


def test_month_end_signal_applies_from_next_trading_day() -> None:
    """A month-end Risk-On signal should not apply until the next trading day."""
    market = _market_frame()
    result = classify_market_regime(market, config=_test_config())

    jan_end = result.loc[result["date"] == pd.Timestamp("2024-01-31")].iloc[0]
    feb_first = result.loc[result["date"] == pd.Timestamp("2024-02-01")].iloc[0]

    assert jan_end["confirmed_regime"] == "Risk-On"
    assert jan_end["regime"] == "Neutral"
    assert feb_first["regime"] == "Risk-On"
    assert feb_first["signal_date"] == pd.Timestamp("2024-01-31")


def test_classification_conditions_and_allocation_columns() -> None:
    """Regime output should include score, condition flags, and equity/cash weights."""
    result = classify_market_regime(_market_frame(), config=_test_config())
    feb = result.loc[result["date"] == pd.Timestamp("2024-02-01")].iloc[0]

    assert bool(feb["kospi_above_ma200"])
    assert bool(feb["momentum_positive"])
    assert bool(feb["breadth_strong"])
    assert int(feb["regime_score"]) > 0
    assert feb["equity_weight"] == 1.0
    assert feb["cash_weight"] == 0.0
    assert get_equity_cash_allocation("Risk-Off") == (0.5, 0.5)


def test_insufficient_data_returns_neutral() -> None:
    """Missing required inputs should produce Neutral rather than a directional regime."""
    market = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "kospi_close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    result = classify_market_regime(market)

    assert set(result["regime"]) == {"Neutral"}
    assert set(result["confirmed_regime"]) == {"Neutral"}
    assert result["regime_score"].eq(0).all()


def test_confirmation_days_reduces_one_day_whipsaw() -> None:
    """Directional raw regimes must persist for confirmation_days before confirmation."""
    market = _manual_market_frame(
        ["Risk-On", "Neutral", "Risk-On", "Risk-On", "Risk-On"],
        start="2024-01-29",
    )
    config = _test_config(confirmation_days=2)

    result = classify_market_regime(market, config=config)

    assert (
        result.loc[result["date"] == pd.Timestamp("2024-01-29"), "confirmed_regime"].iloc[0]
        == "Neutral"
    )
    assert (
        result.loc[result["date"] == pd.Timestamp("2024-02-02"), "confirmed_regime"].iloc[0]
        == "Risk-On"
    )


def test_hysteresis_keeps_previous_regime_inside_buffer() -> None:
    """Hysteresis should retain Risk-On through a small threshold dip."""
    market = _manual_market_frame(["Risk-On", "Neutral", "Risk-On"], start="2024-01-30")
    market.loc[market["date"] == pd.Timestamp("2024-01-31"), "market_breadth"] = 0.53
    config = _test_config(use_hysteresis=True, hysteresis_breadth_buffer=0.04)

    result = classify_market_regime(market, config=config)

    assert (
        result.loc[result["date"] == pd.Timestamp("2024-01-31"), "confirmed_regime"].iloc[0]
        == "Risk-On"
    )


def test_regime_statistics_counts_changes_and_duration() -> None:
    """Statistics should summarize regime transitions and average duration."""
    regimes = pd.DataFrame({"regime": ["Neutral", "Neutral", "Risk-On", "Risk-On", "Risk-Off"]})

    stats = calculate_regime_statistics(regimes)

    assert stats["regime_change_count"] == 2
    assert stats["average_duration"] == 5 / 3
    assert stats["durations"]["Risk-On"] == 2.0


def test_future_data_does_not_change_past_applied_regime() -> None:
    """Changing future market data should not alter already-applied earlier regimes."""
    market = _market_frame()
    baseline = classify_market_regime(market, config=_test_config())

    modified = market.copy()
    future_mask = modified["date"] >= pd.Timestamp("2024-02-15")
    modified.loc[future_mask, "kospi_close"] = 1.0
    modified.loc[future_mask, "kospi_ma200"] = 1000.0
    modified.loc[future_mask, "kospi_momentum_60d"] = -0.5
    modified.loc[future_mask, "market_breadth"] = 0.1
    changed = classify_market_regime(modified, config=_test_config())

    cutoff = baseline["date"] < pd.Timestamp("2024-02-15")
    pd.testing.assert_series_equal(
        baseline.loc[cutoff, "regime"].reset_index(drop=True),
        changed.loc[cutoff, "regime"].reset_index(drop=True),
    )


def _test_config(**overrides: object) -> RegimeConfig:
    values = {
        "kospi_ma_window": 3,
        "kospi_momentum_window": 2,
        "kospi_volatility_window": 2,
        "confirmation_days": 1,
    }
    values.update(overrides)
    return RegimeConfig(**values)


def _market_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2024-02-29")
    close = [100.0 + index for index in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates,
            "kospi_close": close,
            "kospi_ma200": [value - 5.0 for value in close],
            "kospi_momentum_60d": [0.02] * len(dates),
            "kospi_volatility_20d": [0.15] * len(dates),
            "market_breadth": [0.65] * len(dates),
            "kosdaq_close": [200.0 + index for index in range(len(dates))],
        }
    )


def _manual_market_frame(regimes: list[str], *, start: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(regimes))
    rows = []
    for date, regime in zip(dates, regimes, strict=True):
        if regime == "Risk-On":
            rows.append(_manual_row(date, 110.0, 100.0, 0.04, 0.65))
        elif regime == "Risk-Off":
            rows.append(_manual_row(date, 90.0, 100.0, -0.04, 0.35))
        else:
            rows.append(_manual_row(date, 105.0, 100.0, 0.02, 0.50))
    return pd.DataFrame(rows)


def _manual_row(
    date: pd.Timestamp,
    close: float,
    ma200: float,
    momentum: float,
    breadth: float,
) -> dict[str, object]:
    return {
        "date": date,
        "kospi_close": close,
        "kospi_ma200": ma200,
        "kospi_momentum_60d": momentum,
        "kospi_volatility_20d": 0.15,
        "market_breadth": breadth,
    }
