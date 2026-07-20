"""Relative strength factor calculations."""

from __future__ import annotations

import pandas as pd

from src.factors.common import (
    add_grouped_zscore,
    clean_factor_values,
    get_month_end_dates,
    normalize_price_frame,
    trailing_return,
)

MIN_OBSERVATIONS_6M = 127


def calculate_relative_strength(
    prices: pd.DataFrame,
    market_returns: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate monthly relative strength versus market and sector.

    `market_returns` must include `calculation_date`, `market`, and `market_return_6m`.
    `sector_map`, when provided, must include `ticker` and `sector`.
    """
    normalized = normalize_price_frame(prices, {"date", "ticker", "adjusted_close", "market"})
    market_df = market_returns.copy()
    market_df["calculation_date"] = pd.to_datetime(market_df["calculation_date"])
    market_df["market"] = market_df["market"].astype(str)
    calculation_dates = get_month_end_dates(normalized)
    rows: list[dict[str, object]] = []

    sector_lookup = _sector_lookup(sector_map)
    for ticker, ticker_df in normalized.groupby("ticker", sort=True):
        market = str(ticker_df["market"].iloc[-1])
        sector = sector_lookup.get(
            str(ticker), str(ticker_df.get("sector", pd.Series([""])).iloc[-1])
        )
        for calculation_date in calculation_dates:
            stock_return_6m = trailing_return(
                ticker_df,
                calculation_date,
                126,
                min_observations=MIN_OBSERVATIONS_6M,
            )
            market_return_6m = _lookup_market_return(market_df, calculation_date, market)
            rows.append(
                {
                    "calculation_date": calculation_date,
                    "ticker": ticker,
                    "market": market,
                    "sector": sector,
                    "stock_return_6m": stock_return_6m,
                    "market_return_6m": market_return_6m,
                    "market_excess_return": stock_return_6m - market_return_6m,
                }
            )

    result = pd.DataFrame(rows)
    result["sector_median_return_6m"] = result.groupby(["calculation_date", "sector"])[
        "stock_return_6m"
    ].transform("median")
    result["sector_excess_return"] = result["stock_return_6m"] - result["sector_median_return_6m"]
    result["relative_strength_raw"] = (
        0.7 * result["market_excess_return"] + 0.3 * result["sector_excess_return"]
    )
    result = clean_factor_values(result)
    return add_grouped_zscore(result, "relative_strength_raw", "relative_strength_score")


def _lookup_market_return(
    market_df: pd.DataFrame,
    calculation_date: pd.Timestamp,
    market: str,
) -> float:
    matched = market_df.loc[
        (market_df["calculation_date"] == calculation_date) & (market_df["market"] == market),
        "market_return_6m",
    ]
    if matched.empty:
        return float("nan")
    return float(matched.iloc[0])


def _sector_lookup(sector_map: pd.DataFrame | None) -> dict[str, str]:
    if sector_map is None or sector_map.empty:
        return {}
    required_columns = {"ticker", "sector"}
    missing_columns = required_columns - set(sector_map.columns)
    if missing_columns:
        raise ValueError(f"Missing sector map columns: {', '.join(sorted(missing_columns))}")
    normalized = sector_map.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    return dict(zip(normalized["ticker"], normalized["sector"], strict=True))


__all__ = ["calculate_relative_strength"]
