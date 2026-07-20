"""Benchmark portfolio builders."""

from __future__ import annotations

import pandas as pd


def calculate_buy_and_hold_index(
    index_prices: pd.DataFrame,
    *,
    value_column: str,
    benchmark_name: str,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Return buy-and-hold benchmark values from an index price column."""
    required = {"date", value_column}
    missing = required - set(index_prices.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {', '.join(sorted(missing))}")
    result = index_prices.loc[:, ["date", value_column]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").dropna(subset=[value_column]).reset_index(drop=True)
    first = float(result[value_column].iloc[0])
    result["portfolio_value"] = result[value_column].astype(float) / first * initial_value
    result["strategy_name"] = benchmark_name
    return result[["date", "strategy_name", "portfolio_value"]]


def calculate_cash_benchmark(
    dates: pd.Series | list[pd.Timestamp],
    *,
    initial_value: float = 1.0,
    benchmark_name: str = "Cash 100%",
) -> pd.DataFrame:
    """Return a constant cash benchmark."""
    result = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates))})
    result = result.sort_values("date").reset_index(drop=True)
    result["strategy_name"] = benchmark_name
    result["portfolio_value"] = initial_value
    return result


def calculate_equal_weight_universe(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    initial_value: float = 1.0,
    benchmark_name: str = "Universe 100 equal weighted",
) -> pd.DataFrame:
    """Return a simple daily rebalanced equal-weight universe benchmark."""
    required_price = {"date", "ticker", "adjusted_close"}
    missing_price = required_price - set(prices.columns)
    if missing_price:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing_price))}")
    if "ticker" not in universe.columns:
        raise ValueError("universe must contain ticker")

    normalized = prices.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    normalized = normalized.sort_values(["ticker", "date"])
    normalized["daily_return"] = normalized.groupby("ticker")["adjusted_close"].pct_change()
    tickers = set(universe["ticker"].astype(str).str.zfill(6))
    returns = (
        normalized.loc[normalized["ticker"].isin(tickers)]
        .groupby("date")["daily_return"]
        .mean()
        .fillna(0.0)
        .sort_index()
    )
    result = returns.reset_index(name="daily_return")
    result["portfolio_value"] = initial_value * (1.0 + result["daily_return"]).cumprod()
    result["strategy_name"] = benchmark_name
    return result[["date", "strategy_name", "portfolio_value"]]


__all__ = [
    "calculate_buy_and_hold_index",
    "calculate_cash_benchmark",
    "calculate_equal_weight_universe",
]
