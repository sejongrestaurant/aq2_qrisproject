"""Liquidity factor calculations."""

from __future__ import annotations

import pandas as pd

from src.factors.common import (
    add_grouped_zscore,
    clean_factor_values,
    get_month_end_dates,
    normalize_price_frame,
)

MIN_OBSERVATIONS_20D = 20
MIN_OBSERVATIONS_60D = 60


def calculate_liquidity(
    prices: pd.DataFrame,
    *,
    min_median_trading_value: float = 1_000_000_000.0,
    max_zero_volume_ratio: float = 0.2,
) -> pd.DataFrame:
    """Calculate monthly liquidity metrics and eligibility.

    Returns average/median trading value, zero-volume ratio, `liquidity_raw`,
    `liquidity_score`, and `is_liquidity_eligible`.
    """
    normalized = normalize_price_frame(prices, {"date", "ticker", "volume", "trading_value"})
    calculation_dates = get_month_end_dates(normalized)
    rows: list[dict[str, object]] = []

    for ticker, ticker_df in normalized.groupby("ticker", sort=True):
        for calculation_date in calculation_dates:
            history = ticker_df.loc[ticker_df["date"] <= calculation_date].sort_values("date")
            avg_trading_value_20d = _mean_tail(
                history["trading_value"],
                window=20,
                min_observations=MIN_OBSERVATIONS_20D,
            )
            median_trading_value_60d = _median_tail(
                history["trading_value"],
                window=60,
                min_observations=MIN_OBSERVATIONS_60D,
            )
            zero_volume_ratio_60d = _zero_volume_ratio(
                history["volume"],
                window=60,
                min_observations=MIN_OBSERVATIONS_60D,
            )
            is_eligible = (
                pd.notna(median_trading_value_60d)
                and pd.notna(zero_volume_ratio_60d)
                and median_trading_value_60d >= min_median_trading_value
                and zero_volume_ratio_60d <= max_zero_volume_ratio
            )
            liquidity_raw = _combine_liquidity(
                avg_trading_value_20d,
                median_trading_value_60d,
                zero_volume_ratio_60d,
            )
            rows.append(
                {
                    "calculation_date": calculation_date,
                    "ticker": ticker,
                    "avg_trading_value_20d": avg_trading_value_20d,
                    "median_trading_value_60d": median_trading_value_60d,
                    "zero_volume_ratio_60d": zero_volume_ratio_60d,
                    "liquidity_raw": liquidity_raw,
                    "is_liquidity_eligible": bool(is_eligible),
                }
            )

    result = clean_factor_values(pd.DataFrame(rows))
    return add_grouped_zscore(result, "liquidity_raw", "liquidity_score")


def _mean_tail(series: pd.Series, *, window: int, min_observations: int) -> float:
    values = series.tail(window).dropna()
    if len(values) < min_observations:
        return float("nan")
    return float(values.mean())


def _median_tail(series: pd.Series, *, window: int, min_observations: int) -> float:
    values = series.tail(window).dropna()
    if len(values) < min_observations:
        return float("nan")
    return float(values.median())


def _zero_volume_ratio(series: pd.Series, *, window: int, min_observations: int) -> float:
    values = series.tail(window).dropna()
    if len(values) < min_observations:
        return float("nan")
    return float((values == 0).mean())


def _combine_liquidity(
    avg_trading_value_20d: float,
    median_trading_value_60d: float,
    zero_volume_ratio_60d: float,
) -> float:
    if (
        pd.isna(avg_trading_value_20d)
        or pd.isna(median_trading_value_60d)
        or pd.isna(zero_volume_ratio_60d)
    ):
        return float("nan")
    return float(0.5 * avg_trading_value_20d + 0.5 * median_trading_value_60d) * (
        1.0 - zero_volume_ratio_60d
    )


__all__ = ["calculate_liquidity"]
