"""Common helpers for price-based factor calculations.

Input price DataFrame contract:
- Long format with one row per `date` and `ticker`.
- Required columns vary by factor, but `date`, `ticker`, and `adjusted_close` are common.
- `date` can be a Python date, string, or pandas datetime-like value.

Output factor DataFrame contract:
- Long format with one row per monthly `calculation_date` and `ticker`.
- Raw factor columns keep their economic units.
- Score columns are cross-sectional z-scores for each `calculation_date`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_price_frame(df: pd.DataFrame, required_columns: set[str]) -> pd.DataFrame:
    """Return a sorted price DataFrame with normalized dates and tickers."""
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.zfill(6)
    return normalized.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_month_end_dates(df: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Return available month-end calculation dates from observed trading dates."""
    dates = pd.to_datetime(df[date_column], errors="raise")
    month_ends = dates.groupby(dates.dt.to_period("M")).max()
    return month_ends.sort_values().reset_index(drop=True)


def trailing_return(
    ticker_df: pd.DataFrame,
    calculation_date: pd.Timestamp,
    lookback_trading_days: int,
    *,
    skip_recent_trading_days: int = 0,
    min_observations: int,
    price_column: str = "adjusted_close",
) -> float:
    """Calculate trailing return using only prices available by calculation_date."""
    history = ticker_df.loc[ticker_df["date"] <= calculation_date].sort_values("date")
    if skip_recent_trading_days > 0:
        history = history.iloc[:-skip_recent_trading_days]
    needed_observations = lookback_trading_days + 1
    if len(history) < max(min_observations, needed_observations):
        return float("nan")

    end_price = float(history[price_column].iloc[-1])
    start_price = float(history[price_column].iloc[-needed_observations])
    return safe_divide(end_price, start_price) - 1.0


def winsorize_series(
    series: pd.Series, lower_quantile: float = 0.05, upper_quantile: float = 0.95
) -> pd.Series:
    """Winsorize a numeric series by quantile limits."""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_null = numeric.dropna()
    if non_null.empty:
        return numeric
    lower = non_null.quantile(lower_quantile)
    upper = non_null.quantile(upper_quantile)
    return numeric.clip(lower=lower, upper=upper)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Return cross-sectional z-scores with stable handling for small or constant samples."""
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        result = pd.Series(np.nan, index=series.index, dtype="float64")
        result.loc[valid.index] = 0.0
        return result

    return (numeric - valid.mean()) / std


def add_grouped_zscore(
    df: pd.DataFrame,
    raw_column: str,
    score_column: str,
    *,
    invert: bool = False,
) -> pd.DataFrame:
    """Add cross-sectional z-score by calculation date."""
    result = df.copy()
    values = -result[raw_column] if invert else result[raw_column]
    result[score_column] = values.groupby(result["calculation_date"]).transform(
        lambda group: cross_sectional_zscore(winsorize_series(group))
    )
    return result


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide numeric values and return NaN on zero or invalid inputs."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    value = numerator / denominator
    if np.isinf(value):
        return float("nan")
    return float(value)


def clean_factor_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite factor outputs with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "add_grouped_zscore",
    "clean_factor_values",
    "cross_sectional_zscore",
    "get_month_end_dates",
    "normalize_price_frame",
    "safe_divide",
    "trailing_return",
    "winsorize_series",
]
