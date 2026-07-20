"""Low volatility factor calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.common import (
    add_grouped_zscore,
    clean_factor_values,
    get_month_end_dates,
    normalize_price_frame,
)

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS_60D = 60
MIN_OBSERVATIONS_120D = 120
MIN_OBSERVATIONS_252D = 252


def calculate_low_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly low-volatility factor values.

    Lower volatility and lower drawdown produce higher `low_volatility_score`.
    """
    normalized = normalize_price_frame(prices, {"date", "ticker", "adjusted_close"})
    normalized["daily_return"] = normalized.groupby("ticker")["adjusted_close"].pct_change()
    calculation_dates = get_month_end_dates(normalized)
    rows: list[dict[str, object]] = []

    for ticker, ticker_df in normalized.groupby("ticker", sort=True):
        for calculation_date in calculation_dates:
            history = ticker_df.loc[ticker_df["date"] <= calculation_date].sort_values("date")
            volatility_60d = _annualized_volatility(
                history["daily_return"].tail(60), MIN_OBSERVATIONS_60D
            )
            downside_volatility_120d = _annualized_downside_volatility(
                history["daily_return"].tail(120),
                MIN_OBSERVATIONS_120D,
            )
            max_drawdown_252d = _max_drawdown(
                history["adjusted_close"].tail(252),
                MIN_OBSERVATIONS_252D,
            )
            low_volatility_raw = _combine_low_volatility(
                volatility_60d,
                downside_volatility_120d,
                max_drawdown_252d,
            )
            rows.append(
                {
                    "calculation_date": calculation_date,
                    "ticker": ticker,
                    "volatility_60d": volatility_60d,
                    "downside_volatility_120d": downside_volatility_120d,
                    "max_drawdown_252d": max_drawdown_252d,
                    "low_volatility_raw": low_volatility_raw,
                }
            )

    result = clean_factor_values(pd.DataFrame(rows))
    return add_grouped_zscore(result, "low_volatility_raw", "low_volatility_score", invert=True)


def _annualized_volatility(returns: pd.Series, min_observations: int) -> float:
    valid = returns.dropna()
    if len(valid) < min_observations:
        return float("nan")
    return float(valid.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_downside_volatility(returns: pd.Series, min_observations: int) -> float:
    valid = returns.dropna()
    if len(valid) < min_observations:
        return float("nan")
    downside = valid.loc[valid < 0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(prices: pd.Series, min_observations: int) -> float:
    valid = prices.dropna()
    if len(valid) < min_observations:
        return float("nan")
    running_max = valid.cummax()
    drawdowns = valid / running_max - 1.0
    return float(drawdowns.min())


def _combine_low_volatility(
    volatility_60d: float,
    downside_volatility_120d: float,
    max_drawdown_252d: float,
) -> float:
    if pd.isna(volatility_60d) or pd.isna(downside_volatility_120d) or pd.isna(max_drawdown_252d):
        return float("nan")
    return float(
        0.4 * volatility_60d + 0.3 * downside_volatility_120d + 0.3 * abs(max_drawdown_252d)
    )


__all__ = ["calculate_low_volatility"]
