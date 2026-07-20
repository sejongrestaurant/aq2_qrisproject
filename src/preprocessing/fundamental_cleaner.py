"""Normalize DART financial statement rows for factor research."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Final

import pandas as pd

ACCOUNT_NAME_MAP: Final[dict[str, tuple[str, ...]]] = {
    "revenue": ("매출액", "영업수익", "수익(매출액)", "매출"),
    "operating_income": ("영업이익", "영업손실"),
    "net_income": ("당기순이익", "당기순손실", "분기순이익", "반기순이익"),
    "total_assets": ("자산총계",),
    "total_equity": ("자본총계",),
    "total_debt": ("부채총계",),
    "operating_cash_flow": ("영업활동현금흐름", "영업활동으로 인한 현금흐름"),
    "shares_outstanding": ("발행주식수", "보통주식수"),
}
FLOW_FIELDS: Final[frozenset[str]] = frozenset(
    {"revenue", "operating_income", "net_income", "operating_cash_flow"}
)
REPORT_CODE_TO_QUARTER: Final[dict[str, int]] = {
    "11013": 1,
    "11012": 2,
    "11014": 3,
    "11011": 4,
}


def clean_dart_fundamentals(
    raw_df: pd.DataFrame,
    *,
    ticker: str,
    fiscal_year: int,
    report_code: str,
    report_date: date,
    available_date: date,
) -> tuple[dict[str, object], list[str]]:
    """Convert raw DART account rows into one database-ready fundamental record."""
    values: dict[str, object] = {
        "ticker": str(ticker).zfill(6),
        "report_date": report_date,
        "available_date": available_date,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": REPORT_CODE_TO_QUARTER[report_code],
    }
    account_values = _extract_account_values(raw_df)
    missing_accounts = sorted(set(ACCOUNT_NAME_MAP) - set(account_values))

    for field in ACCOUNT_NAME_MAP:
        values[field] = account_values.get(field)
    return values, missing_accounts


def convert_cumulative_quarters_to_single_period(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative quarterly flow fields into single-quarter values."""
    if df.empty:
        return df.copy()

    required_columns = {"ticker", "fiscal_year", "fiscal_quarter"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    result = df.sort_values(["ticker", "fiscal_year", "fiscal_quarter"]).copy()
    for field in FLOW_FIELDS & set(result.columns):
        previous = result.groupby(["ticker", "fiscal_year"], sort=False)[field].shift(1)
        result[field] = result[field].where(result["fiscal_quarter"] == 1, result[field] - previous)
    return result


def infer_report_date(fiscal_year: int, report_code: str) -> date:
    """Infer fiscal period end date from DART report code."""
    quarter = REPORT_CODE_TO_QUARTER[report_code]
    if quarter == 1:
        return date(fiscal_year, 3, 31)
    if quarter == 2:
        return date(fiscal_year, 6, 30)
    if quarter == 3:
        return date(fiscal_year, 9, 30)
    return date(fiscal_year, 12, 31)


def _extract_account_values(raw_df: pd.DataFrame) -> dict[str, Decimal]:
    account_values: dict[str, Decimal] = {}
    if raw_df.empty:
        return account_values

    for _, row in raw_df.iterrows():
        account_name = str(row.get("account_nm", "")).strip()
        field = _map_account_name(account_name)
        if field is None or field in account_values:
            continue
        account_values[field] = _parse_decimal(row.get("thstrm_amount"))
    return account_values


def _map_account_name(account_name: str) -> str | None:
    normalized = account_name.replace(" ", "")
    for field, aliases in ACCOUNT_NAME_MAP.items():
        if any(alias.replace(" ", "") == normalized for alias in aliases):
            return field
    return None


def _parse_decimal(value: object) -> Decimal:
    text = str(value).replace(",", "").replace(" ", "").strip()
    if text in {"", "-", "nan", "None"}:
        return Decimal("0")
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Cannot parse DART numeric amount: {value}") from error


__all__ = [
    "ACCOUNT_NAME_MAP",
    "FLOW_FIELDS",
    "REPORT_CODE_TO_QUARTER",
    "clean_dart_fundamentals",
    "convert_cumulative_quarters_to_single_period",
    "infer_report_date",
]
