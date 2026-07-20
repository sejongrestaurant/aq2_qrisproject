"""Tests for quality and growth factor calculations."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from src.factors.growth import calculate_growth
from src.factors.quality import calculate_quality


def test_quality_uses_average_equity_and_assets() -> None:
    """ROE and ROA should use average balance-sheet denominators."""
    fundamentals = _fundamentals()

    result = calculate_quality(fundamentals, [pd.Timestamp("2024-05-20")])
    row = result.loc[result["ticker"] == "005930"].iloc[0]

    assert math.isclose(row["roe"], 120.0 / ((900.0 + 1000.0) / 2))
    assert math.isclose(row["roa"], 120.0 / ((1900.0 + 2000.0) / 2))
    assert "quality_score" in result.columns
    assert "quality_sector_score" in result.columns


def test_quality_sets_nan_for_non_positive_equity() -> None:
    """Companies with non-positive equity should not receive fabricated ROE/debt scores."""
    fundamentals = _fundamentals()
    fundamentals.loc[fundamentals["ticker"] == "000660", "total_equity"] = 0.0

    result = calculate_quality(fundamentals, [pd.Timestamp("2024-05-20")])
    row = result.loc[result["ticker"] == "000660"].iloc[0]

    assert pd.isna(row["roe"])
    assert pd.isna(row["debt_ratio"])


def test_growth_detects_turnaround_and_deterioration() -> None:
    """Sign changes should use flags instead of misleading growth rates."""
    fundamentals = _fundamentals()

    result = calculate_growth(fundamentals, [pd.Timestamp("2024-05-20")])
    turnaround = result.loc[result["ticker"] == "005930"].iloc[0]
    deterioration = result.loc[result["ticker"] == "000660"].iloc[0]

    assert bool(turnaround["operating_income_turnaround"]) is True
    assert pd.isna(turnaround["operating_income_growth_yoy"])
    assert bool(deterioration["net_income_deterioration"]) is True
    assert pd.isna(deterioration["net_income_growth_yoy"])


def test_growth_calculates_revenue_yoy_and_cagr() -> None:
    """Growth should calculate same-quarter revenue growth and 3-year annual revenue CAGR."""
    fundamentals = _fundamentals()

    result = calculate_growth(fundamentals, [pd.Timestamp("2024-05-20")])
    row = result.loc[result["ticker"] == "035720"].iloc[0]

    assert math.isclose(row["revenue_growth_yoy"], 0.2)
    assert row["revenue_cagr_3y"] > 0
    assert "growth_score" in result.columns
    assert "growth_sector_score" in result.columns


def test_fundamental_factors_do_not_use_future_available_data() -> None:
    """Changing a future filing should not alter a factor before its available_date."""
    fundamentals = _fundamentals()
    baseline = calculate_quality(fundamentals, [pd.Timestamp("2024-05-20")])

    modified = fundamentals.copy()
    future_mask = (modified["ticker"] == "005930") & (
        pd.to_datetime(modified["available_date"]) > pd.Timestamp("2024-05-20")
    )
    modified.loc[future_mask, "net_income"] = 999999.0
    changed = calculate_quality(modified, [pd.Timestamp("2024-05-20")])

    baseline_roe = baseline.loc[baseline["ticker"] == "005930", "roe"].iloc[0]
    changed_roe = changed.loc[changed["ticker"] == "005930", "roe"].iloc[0]

    assert baseline_roe == changed_roe


def test_nan_policy_fill_median_can_produce_composite_when_one_component_missing() -> None:
    """NaN handling policy should be selectable by caller."""
    fundamentals = _fundamentals()
    fundamentals.loc[fundamentals["ticker"] == "035720", "revenue"] = 0.0

    propagate = calculate_quality(
        fundamentals,
        [pd.Timestamp("2024-05-20")],
        nan_policy="propagate",
    )
    filled = calculate_quality(
        fundamentals,
        [pd.Timestamp("2024-05-20")],
        nan_policy="fill_median",
    )

    assert propagate["quality_raw"].isna().any()
    assert filled["quality_raw"].notna().any()


def _fundamentals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        _ticker_rows(
            ticker="005930",
            sector="Technology",
            current_revenue=1200.0,
            previous_revenue=1000.0,
            current_operating_income=100.0,
            previous_operating_income=-10.0,
            current_net_income=120.0,
            previous_net_income=100.0,
            current_assets=2000.0,
            previous_assets=1900.0,
            current_equity=1000.0,
            previous_equity=900.0,
        )
    )
    rows.extend(
        _ticker_rows(
            ticker="000660",
            sector="Technology",
            current_revenue=900.0,
            previous_revenue=1000.0,
            current_operating_income=80.0,
            previous_operating_income=100.0,
            current_net_income=-20.0,
            previous_net_income=80.0,
            current_assets=1800.0,
            previous_assets=1700.0,
            current_equity=800.0,
            previous_equity=780.0,
        )
    )
    rows.extend(
        _ticker_rows(
            ticker="035720",
            sector="Internet",
            current_revenue=600.0,
            previous_revenue=500.0,
            current_operating_income=90.0,
            previous_operating_income=75.0,
            current_net_income=70.0,
            previous_net_income=60.0,
            current_assets=1000.0,
            previous_assets=900.0,
            current_equity=700.0,
            previous_equity=650.0,
        )
    )
    return pd.DataFrame(rows)


def _ticker_rows(
    *,
    ticker: str,
    sector: str,
    current_revenue: float,
    previous_revenue: float,
    current_operating_income: float,
    previous_operating_income: float,
    current_net_income: float,
    previous_net_income: float,
    current_assets: float,
    previous_assets: float,
    current_equity: float,
    previous_equity: float,
) -> list[dict[str, object]]:
    rows = [
        _row(
            ticker=ticker,
            sector=sector,
            report_date=date(2023, 3, 31),
            available_date=date(2023, 5, 15),
            fiscal_year=2023,
            fiscal_quarter=1,
            revenue=previous_revenue,
            operating_income=previous_operating_income,
            net_income=previous_net_income,
            total_assets=previous_assets,
            total_equity=previous_equity,
        ),
        _row(
            ticker=ticker,
            sector=sector,
            report_date=date(2024, 3, 31),
            available_date=date(2024, 5, 15),
            fiscal_year=2024,
            fiscal_quarter=1,
            revenue=current_revenue,
            operating_income=current_operating_income,
            net_income=current_net_income,
            total_assets=current_assets,
            total_equity=current_equity,
        ),
        _row(
            ticker=ticker,
            sector=sector,
            report_date=date(2024, 6, 30),
            available_date=date(2024, 8, 15),
            fiscal_year=2024,
            fiscal_quarter=2,
            revenue=current_revenue * 2,
            operating_income=current_operating_income * 2,
            net_income=current_net_income * 2,
            total_assets=current_assets * 1.1,
            total_equity=current_equity * 1.1,
        ),
    ]
    for fiscal_year, revenue in [
        (2020, 350.0),
        (2021, 400.0),
        (2022, 450.0),
        (2023, 500.0),
        (2024, 600.0),
    ]:
        rows.append(
            _row(
                ticker=ticker,
                sector=sector,
                report_date=date(fiscal_year, 12, 31),
                available_date=date(fiscal_year + 1, 3, 20),
                fiscal_year=fiscal_year,
                fiscal_quarter=4,
                revenue=revenue,
                operating_income=revenue * 0.1,
                net_income=revenue * 0.08,
                total_assets=current_assets,
                total_equity=current_equity,
            )
        )
    return rows


def _row(
    *,
    ticker: str,
    sector: str,
    report_date: date,
    available_date: date,
    fiscal_year: int,
    fiscal_quarter: int,
    revenue: float,
    operating_income: float,
    net_income: float,
    total_assets: float,
    total_equity: float,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "sector": sector,
        "report_date": report_date,
        "available_date": available_date,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "revenue": revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "total_assets": total_assets,
        "total_equity": total_equity,
        "total_debt": total_assets - total_equity,
        "operating_cash_flow": net_income * 1.2,
        "shares_outstanding": 1000.0,
    }
