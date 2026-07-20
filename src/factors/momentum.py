"""Momentum factor calculations."""

from __future__ import annotations

import pandas as pd

from src.factors.common import (
    add_grouped_zscore,
    clean_factor_values,
    get_month_end_dates,
    normalize_price_frame,
    trailing_return,
)

MIN_OBSERVATIONS_12_1 = 253
MIN_OBSERVATIONS_6M = 127
MIN_OBSERVATIONS_3M = 64


def calculate_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly 12-1, 6M, and 3M momentum factor values.

    Returns columns: `calculation_date`, `ticker`, `momentum_12_1`, `momentum_6m`,
    `momentum_3m`, `momentum_raw`, and `momentum_score`.
    """
    normalized = normalize_price_frame(prices, {"date", "ticker", "adjusted_close"})
    calculation_dates = get_month_end_dates(normalized)
    rows: list[dict[str, object]] = []

    for ticker, ticker_df in normalized.groupby("ticker", sort=True):
        for calculation_date in calculation_dates:
            momentum_12_1 = trailing_return(
                ticker_df,
                calculation_date,
                252,
                skip_recent_trading_days=21,
                min_observations=MIN_OBSERVATIONS_12_1 + 21,
            )
            momentum_6m = trailing_return(
                ticker_df,
                calculation_date,
                126,
                min_observations=MIN_OBSERVATIONS_6M,
            )
            momentum_3m = trailing_return(
                ticker_df,
                calculation_date,
                63,
                min_observations=MIN_OBSERVATIONS_3M,
            )
            momentum_raw = 0.5 * momentum_12_1 + 0.3 * momentum_6m + 0.2 * momentum_3m
            rows.append(
                {
                    "calculation_date": calculation_date,
                    "ticker": ticker,
                    "momentum_12_1": momentum_12_1,
                    "momentum_6m": momentum_6m,
                    "momentum_3m": momentum_3m,
                    "momentum_raw": momentum_raw,
                }
            )

    result = clean_factor_values(pd.DataFrame(rows))
    return add_grouped_zscore(result, "momentum_raw", "momentum_score")


__all__ = ["calculate_momentum"]
