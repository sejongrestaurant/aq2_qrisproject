"""Growth factor calculations from point-in-time fundamentals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.common import cross_sectional_zscore, safe_divide, winsorize_series
from src.factors.quality import _normalize_fundamentals

GROWTH_WEIGHTS = {
    "revenue_growth_yoy": 0.35,
    "operating_income_growth_yoy": 0.35,
    "net_income_growth_yoy": 0.20,
    "revenue_cagr_3y": 0.10,
}


def calculate_growth(
    fundamentals: pd.DataFrame,
    calculation_dates: list[pd.Timestamp] | pd.Series,
    *,
    sector_map: pd.DataFrame | None = None,
    nan_policy: str = "propagate",
) -> pd.DataFrame:
    """Calculate point-in-time growth factors.

    For each `calculation_date`, only rows with `available_date <= calculation_date` are visible.
    The latest available report per ticker is compared against the same fiscal quarter in the
    previous fiscal year. If same-quarter data is unavailable, the row is left as NaN rather than
    filled with zero. Revenue CAGR uses annual (`fiscal_quarter == 4`) revenue from the latest
    fiscal year and three fiscal years earlier.

    Growth formulas use `(current / previous) - 1` when both values are positive. If operating
    income or net income moves from loss/non-positive to profit, the corresponding turnaround flag
    is set and the numeric growth rate is NaN. If it moves from profit to loss/non-positive, the
    deterioration flag is set and the numeric growth rate is NaN. This avoids misleading infinite
    growth rates around sign changes.

    `nan_policy="propagate"` leaves the composite NaN when a weighted component is unavailable.
    `nan_policy="fill_median"` fills component raw values with the calculation-date median before
    combining. The result includes raw metrics, full-market z-scores, and sector-neutral z-scores.
    """
    _validate_nan_policy(nan_policy)
    normalized = _normalize_fundamentals(fundamentals, sector_map)
    rows: list[dict[str, object]] = []

    for calculation_date in pd.to_datetime(calculation_dates):
        available = normalized.loc[normalized["available_date"] <= calculation_date]
        for ticker, ticker_df in available.groupby("ticker", sort=True):
            sorted_df = ticker_df.sort_values(["available_date", "report_date"])
            latest = sorted_df.iloc[-1]
            previous = _same_quarter_previous_year(sorted_df, latest)
            row = _growth_snapshot(sorted_df, latest, previous)
            row["calculation_date"] = calculation_date
            row["ticker"] = ticker
            row["sector"] = _latest_sector(sorted_df)
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = _add_growth_raw(result, nan_policy)
    return _add_scores(result, "growth_raw", "growth_score", "growth_sector_score")


def _growth_snapshot(
    ticker_df: pd.DataFrame,
    latest: pd.Series,
    previous: pd.Series | None,
) -> dict[str, object]:
    if previous is None:
        return {
            "revenue_growth_yoy": float("nan"),
            "operating_income_growth_yoy": float("nan"),
            "net_income_growth_yoy": float("nan"),
            "revenue_cagr_3y": _revenue_cagr_3y(ticker_df, latest),
            "operating_income_turnaround": False,
            "operating_income_deterioration": False,
            "net_income_turnaround": False,
            "net_income_deterioration": False,
        }

    operating_growth, operating_turnaround, operating_deterioration = _signed_growth(
        float(latest["operating_income"]), float(previous["operating_income"])
    )
    net_growth, net_turnaround, net_deterioration = _signed_growth(
        float(latest["net_income"]), float(previous["net_income"])
    )
    return {
        "revenue_growth_yoy": _positive_base_growth(
            float(latest["revenue"]), float(previous["revenue"])
        ),
        "operating_income_growth_yoy": operating_growth,
        "net_income_growth_yoy": net_growth,
        "revenue_cagr_3y": _revenue_cagr_3y(ticker_df, latest),
        "operating_income_turnaround": operating_turnaround,
        "operating_income_deterioration": operating_deterioration,
        "net_income_turnaround": net_turnaround,
        "net_income_deterioration": net_deterioration,
    }


def _same_quarter_previous_year(ticker_df: pd.DataFrame, latest: pd.Series) -> pd.Series | None:
    previous = ticker_df.loc[
        (ticker_df["fiscal_year"] == latest["fiscal_year"] - 1)
        & (ticker_df["fiscal_quarter"] == latest["fiscal_quarter"])
    ].sort_values(["available_date", "report_date"])
    if previous.empty:
        return None
    return previous.iloc[-1]


def _positive_base_growth(current: float, previous: float) -> float:
    if current <= 0 or previous <= 0:
        return float("nan")
    return safe_divide(current, previous) - 1.0


def _signed_growth(current: float, previous: float) -> tuple[float, bool, bool]:
    turnaround = previous <= 0 < current
    deterioration = previous > 0 >= current
    if turnaround or deterioration or previous <= 0 or current <= 0:
        return float("nan"), turnaround, deterioration
    return safe_divide(current, previous) - 1.0, False, False


def _revenue_cagr_3y(ticker_df: pd.DataFrame, latest: pd.Series) -> float:
    annual = ticker_df.loc[ticker_df["fiscal_quarter"] == 4].sort_values(
        ["fiscal_year", "available_date"]
    )
    if annual.empty:
        return float("nan")
    current_year = int(latest["fiscal_year"])
    current_candidates = annual.loc[annual["fiscal_year"] <= current_year]
    if current_candidates.empty:
        return float("nan")
    current = current_candidates.iloc[-1]
    base_candidates = annual.loc[annual["fiscal_year"] == int(current["fiscal_year"]) - 3]
    if base_candidates.empty:
        return float("nan")
    base = base_candidates.iloc[-1]
    if current["revenue"] <= 0 or base["revenue"] <= 0:
        return float("nan")
    return float((current["revenue"] / base["revenue"]) ** (1 / 3) - 1)


def _add_growth_raw(df: pd.DataFrame, nan_policy: str) -> pd.DataFrame:
    result = df.copy()
    if nan_policy == "fill_median":
        for column in GROWTH_WEIGHTS:
            result[column] = result.groupby("calculation_date")[column].transform(
                lambda group: group.fillna(group.median())
            )
    result["growth_raw"] = sum(result[column] * weight for column, weight in GROWTH_WEIGHTS.items())
    return result.replace([np.inf, -np.inf], np.nan)


def _add_scores(
    df: pd.DataFrame,
    raw_column: str,
    score_column: str,
    sector_score_column: str,
) -> pd.DataFrame:
    result = df.copy()
    result[score_column] = result.groupby("calculation_date")[raw_column].transform(
        lambda group: cross_sectional_zscore(winsorize_series(group))
    )
    result[sector_score_column] = result.groupby(["calculation_date", "sector"])[
        raw_column
    ].transform(lambda group: cross_sectional_zscore(winsorize_series(group)))
    return result


def _latest_sector(ticker_df: pd.DataFrame) -> str:
    sector = ticker_df["sector"].dropna()
    return str(sector.iloc[-1]) if not sector.empty else "Unknown"


def _validate_nan_policy(nan_policy: str) -> None:
    if nan_policy not in {"propagate", "fill_median"}:
        raise ValueError("nan_policy must be either 'propagate' or 'fill_median'")


__all__ = ["calculate_growth"]
