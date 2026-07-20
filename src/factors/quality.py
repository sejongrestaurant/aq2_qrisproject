"""Quality factor calculations from point-in-time fundamentals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.common import cross_sectional_zscore, safe_divide, winsorize_series

QUALITY_WEIGHTS = {
    "roe": 0.30,
    "operating_margin": 0.20,
    "operating_cash_flow_ratio": 0.20,
    "roa": 0.15,
    "low_debt_ratio": 0.15,
}
FLOW_COLUMNS = ["revenue", "operating_income", "net_income", "operating_cash_flow"]


def calculate_quality(
    fundamentals: pd.DataFrame,
    calculation_dates: list[pd.Timestamp] | pd.Series,
    *,
    sector_map: pd.DataFrame | None = None,
    nan_policy: str = "propagate",
) -> pd.DataFrame:
    """Calculate point-in-time quality factors.

    The input must be long format with one row per ticker and financial reporting period.
    Required columns are `ticker`, `report_date`, `available_date`, `fiscal_year`,
    `fiscal_quarter`, `revenue`, `operating_income`, `net_income`, `total_assets`,
    `total_equity`, `total_debt`, and `operating_cash_flow`. A `sector` column is optional;
    alternatively pass `sector_map` with `ticker` and `sector`.

    For every `calculation_date`, the function first filters to rows whose
    `available_date <= calculation_date`. Later filings and corrections are invisible, which
    prevents look-ahead bias. Flow items use trailing-twelve-month sums when at least four
    quarterly rows are available; otherwise the latest annual row (`fiscal_quarter == 4`) is used.
    Balance sheet denominators use the average of the latest and previous available balances.

    Formulas:
    - ROE = TTM net income / average equity.
    - ROA = TTM net income / average assets.
    - operating margin = TTM operating income / TTM revenue.
    - operating cash flow ratio = TTM operating cash flow / average assets.
    - debt ratio = total debt / total equity. Lower debt receives a higher contribution via
      `low_debt_ratio = -debt_ratio`.

    If equity is zero or negative, ROE and debt ratio are set to NaN rather than clipped or
    assigned an arbitrary score. `nan_policy="propagate"` keeps missing components as NaN in the
    composite. `nan_policy="fill_median"` fills component raw values with the calculation-date
    cross-sectional median before combining. Raw values, full-market z-scores, and sector-neutral
    z-scores are all returned.
    """
    _validate_nan_policy(nan_policy)
    normalized = _normalize_fundamentals(fundamentals, sector_map)
    rows: list[dict[str, object]] = []

    for calculation_date in pd.to_datetime(calculation_dates):
        available = normalized.loc[normalized["available_date"] <= calculation_date]
        for ticker, ticker_df in available.groupby("ticker", sort=True):
            snapshot = _quality_snapshot(ticker_df)
            snapshot["calculation_date"] = calculation_date
            snapshot["ticker"] = ticker
            snapshot["sector"] = _latest_sector(ticker_df)
            rows.append(snapshot)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = _add_quality_raw(result, nan_policy)
    return _add_scores(result, "quality_raw", "quality_score", "quality_sector_score")


def _quality_snapshot(ticker_df: pd.DataFrame) -> dict[str, float]:
    latest = ticker_df.sort_values(["available_date", "report_date"]).iloc[-1]
    flow_values = _ttm_or_annual_flows(ticker_df)
    same_quarter_previous_year = ticker_df.loc[
        (ticker_df["fiscal_year"] == latest["fiscal_year"] - 1)
        & (ticker_df["fiscal_quarter"] == latest["fiscal_quarter"])
    ].sort_values(["available_date", "report_date"])
    earlier_balance = ticker_df.loc[ticker_df["report_date"] < latest["report_date"]].sort_values(
        ["available_date", "report_date"]
    )
    if not same_quarter_previous_year.empty:
        previous = same_quarter_previous_year.iloc[-1]
    elif not earlier_balance.empty:
        previous = earlier_balance.iloc[-1]
    else:
        previous = latest

    avg_assets = np.nanmean([latest["total_assets"], previous["total_assets"]])
    avg_equity = np.nanmean([latest["total_equity"], previous["total_equity"]])
    equity = float(latest["total_equity"])
    debt_ratio = safe_divide(float(latest["total_debt"]), equity) if equity > 0 else float("nan")

    return {
        "roe": safe_divide(float(flow_values["net_income"]), float(avg_equity))
        if avg_equity > 0
        else float("nan"),
        "roa": safe_divide(float(flow_values["net_income"]), float(avg_assets)),
        "operating_margin": safe_divide(
            float(flow_values["operating_income"]), float(flow_values["revenue"])
        ),
        "operating_cash_flow_ratio": safe_divide(
            float(flow_values["operating_cash_flow"]), float(avg_assets)
        ),
        "debt_ratio": debt_ratio,
        "low_debt_ratio": -debt_ratio if pd.notna(debt_ratio) else float("nan"),
    }


def _ttm_or_annual_flows(ticker_df: pd.DataFrame) -> pd.Series:
    sorted_df = ticker_df.sort_values(["report_date", "available_date"])
    latest = sorted_df.iloc[-1]
    current_year_quarters = sorted_df.loc[
        (sorted_df["fiscal_year"] == latest["fiscal_year"]) & (sorted_df["fiscal_quarter"] < 4)
    ]
    if not current_year_quarters.empty and len(current_year_quarters) < 4:
        return latest[FLOW_COLUMNS]

    latest_four = sorted_df.tail(4)
    if len(latest_four) >= 4:
        return latest_four[FLOW_COLUMNS].sum(min_count=4)

    annual_rows = sorted_df.loc[sorted_df["fiscal_quarter"] == 4]
    if not annual_rows.empty:
        return annual_rows.iloc[-1][FLOW_COLUMNS]
    return sorted_df.iloc[-1][FLOW_COLUMNS] * np.nan


def _add_quality_raw(df: pd.DataFrame, nan_policy: str) -> pd.DataFrame:
    result = df.copy()
    component_columns = list(QUALITY_WEIGHTS)
    if nan_policy == "fill_median":
        for column in component_columns:
            result[column] = result.groupby("calculation_date")[column].transform(
                lambda group: group.fillna(group.median())
            )

    result["quality_raw"] = sum(
        result[column] * weight for column, weight in QUALITY_WEIGHTS.items()
    )
    return result


def _normalize_fundamentals(
    fundamentals: pd.DataFrame, sector_map: pd.DataFrame | None
) -> pd.DataFrame:
    required = {
        "ticker",
        "report_date",
        "available_date",
        "fiscal_year",
        "fiscal_quarter",
        "revenue",
        "operating_income",
        "net_income",
        "total_assets",
        "total_equity",
        "total_debt",
        "operating_cash_flow",
    }
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    normalized = fundamentals.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    normalized["report_date"] = pd.to_datetime(normalized["report_date"])
    normalized["available_date"] = pd.to_datetime(normalized["available_date"])
    for column in required - {"ticker", "report_date", "available_date"}:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if sector_map is not None:
        normalized = normalized.merge(_normalize_sector_map(sector_map), on="ticker", how="left")
    if "sector" not in normalized.columns:
        normalized["sector"] = "Unknown"
    return normalized.sort_values(["ticker", "available_date", "report_date"])


def _normalize_sector_map(sector_map: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "sector"}
    missing = required - set(sector_map.columns)
    if missing:
        raise ValueError(f"Missing sector map columns: {', '.join(sorted(missing))}")
    normalized = sector_map.loc[:, ["ticker", "sector"]].copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    return normalized.drop_duplicates("ticker", keep="last")


def _latest_sector(ticker_df: pd.DataFrame) -> str:
    sector = ticker_df["sector"].dropna()
    return str(sector.iloc[-1]) if not sector.empty else "Unknown"


def _add_scores(
    df: pd.DataFrame,
    raw_column: str,
    score_column: str,
    sector_score_column: str,
) -> pd.DataFrame:
    result = df.replace([np.inf, -np.inf], np.nan).copy()
    result[score_column] = result.groupby("calculation_date")[raw_column].transform(
        lambda group: cross_sectional_zscore(winsorize_series(group))
    )
    result[sector_score_column] = result.groupby(["calculation_date", "sector"])[
        raw_column
    ].transform(lambda group: cross_sectional_zscore(winsorize_series(group)))
    return result


def _validate_nan_policy(nan_policy: str) -> None:
    if nan_policy not in {"propagate", "fill_median"}:
        raise ValueError("nan_policy must be either 'propagate' or 'fill_median'")


__all__ = ["calculate_quality"]
