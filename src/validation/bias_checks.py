"""Look-ahead and backtest bias checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Severity = Literal["critical", "warning", "info"]


@dataclass(frozen=True)
class AuditIssue:
    """One validation finding for audit reports."""

    severity: Severity
    check_name: str
    passed: bool
    affected_dates: list[str]
    affected_tickers: list[str]
    message: str
    suggested_fix: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly dict."""
        return {
            "severity": self.severity,
            "check_name": self.check_name,
            "passed": self.passed,
            "affected_dates": self.affected_dates,
            "affected_tickers": self.affected_tickers,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


def passed_issue(check_name: str, message: str) -> AuditIssue:
    """Create a passed audit issue row."""
    return AuditIssue(
        severity="info",
        check_name=check_name,
        passed=True,
        affected_dates=[],
        affected_tickers=[],
        message=message,
        suggested_fix="No action required.",
    )


def check_prices_before_listing(prices: pd.DataFrame, universe: pd.DataFrame) -> AuditIssue:
    """Detect price rows dated before each ticker's listing_date or data_start_date."""
    check = "prices_before_listing"
    required_prices = {"date", "ticker"}
    required_universe = {"ticker"}
    if not required_prices <= set(prices.columns) or not required_universe <= set(universe.columns):
        return _failed_schema(check, "prices requires date/ticker and universe requires ticker")

    date_column = "listing_date" if "listing_date" in universe.columns else "data_start_date"
    if date_column not in universe.columns:
        return _failed_schema(check, "universe requires listing_date or data_start_date")

    price_df = _normalize_ticker_date(prices, "date")
    universe_df = universe.copy()
    universe_df["ticker"] = universe_df["ticker"].astype(str).str.zfill(6)
    universe_df[date_column] = pd.to_datetime(universe_df[date_column])
    merged = price_df.merge(universe_df[["ticker", date_column]], on="ticker", how="left")
    bad = merged.loc[merged["date"] < merged[date_column]]
    return _issue_from_bad_rows(
        check,
        bad,
        "critical",
        "Price rows exist before a stock was actually listed.",
        "Drop pre-listing prices and enforce point-in-time listing filters before backtests.",
    )


def check_fundamentals_available_date_usage(factor_scores: pd.DataFrame) -> AuditIssue:
    """Detect factor rows whose available_date is after calculation_date."""
    check = "fundamentals_available_date_usage"
    required = {"calculation_date", "ticker", "available_date"}
    if not required <= set(factor_scores.columns):
        return _failed_schema(
            check, "factor_scores requires calculation_date, ticker, and available_date"
        )

    df = _normalize_ticker_date(factor_scores, "calculation_date")
    df["available_date"] = pd.to_datetime(df["available_date"])
    bad = df.loc[df["available_date"] > df["calculation_date"]]
    return _issue_from_bad_rows(
        check,
        bad.rename(columns={"calculation_date": "date"}),
        "critical",
        "Factor rows use fundamentals before their available_date.",
        "Filter factors with available_date <= signal/calculation date.",
    )


def check_same_day_signal_execution(trades: pd.DataFrame) -> AuditIssue:
    """Detect trades executed on the same date as their rebalance signal."""
    check = "same_day_signal_execution"
    required = {"signal_date", "execution_date", "ticker"}
    if not required <= set(trades.columns):
        return _failed_schema(check, "trades requires signal_date, execution_date, and ticker")

    df = trades.copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    bad = df.loc[df["signal_date"] >= df["execution_date"]].rename(
        columns={"execution_date": "date"}
    )
    return _issue_from_bad_rows(
        check,
        bad,
        "critical",
        "Trades are executed on or before the signal date.",
        "Execute month-end signals on the next trading day or later.",
    )


def check_future_universe_membership(
    selections: pd.DataFrame, universe: pd.DataFrame
) -> AuditIssue:
    """Detect selected holdings before point-in-time universe membership was valid."""
    check = "future_universe_membership"
    required_selections = {"date", "ticker"}
    required_universe = {"ticker"}
    if not required_selections <= set(selections.columns) or not required_universe <= set(
        universe.columns
    ):
        return _failed_schema(check, "selections requires date/ticker and universe requires ticker")
    start_column = (
        "membership_start_date" if "membership_start_date" in universe.columns else "listing_date"
    )
    if start_column not in universe.columns:
        return _failed_schema(check, "universe requires membership_start_date or listing_date")

    selected = _normalize_ticker_date(selections, "date")
    universe_df = universe.copy()
    universe_df["ticker"] = universe_df["ticker"].astype(str).str.zfill(6)
    universe_df[start_column] = pd.to_datetime(universe_df[start_column])
    merged = selected.merge(universe_df[["ticker", start_column]], on="ticker", how="left")
    bad = merged.loc[merged["date"] < merged[start_column]]
    return _issue_from_bad_rows(
        check,
        bad,
        "critical",
        "Selections include stocks before valid universe membership.",
        "Build the candidate universe using point-in-time membership for each signal date.",
    )


def check_backfilled_missing_data(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    ticker_column: str = "ticker",
    source_flag_column: str = "filled_from_future",
) -> AuditIssue:
    """Detect explicit future backfill flags in cleaned data."""
    check = "future_backfill"
    if source_flag_column not in data.columns:
        return AuditIssue(
            severity="warning",
            check_name=check,
            passed=False,
            affected_dates=[],
            affected_tickers=[],
            message="No filled_from_future audit flag was found; future backfill cannot be ruled out.",
            suggested_fix="Keep fill direction metadata during preprocessing and avoid bfill in time series.",
        )
    df = data.copy()
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    bad = df.loc[df[source_flag_column].fillna(False).astype(bool)]
    if date_column != "date" and date_column in bad.columns:
        bad = bad.rename(columns={date_column: "date"})
    if ticker_column != "ticker" and ticker_column in bad.columns:
        bad = bad.rename(columns={ticker_column: "ticker"})
    return _issue_from_bad_rows(
        check,
        bad,
        "critical",
        "Data rows are flagged as filled from future observations.",
        "Replace backfill with forward-fill from already observed values or leave values missing.",
    )


def check_survivorship_bias_risk(universe: pd.DataFrame) -> AuditIssue:
    """Warn when universe lacks delisting or point-in-time membership fields."""
    check = "survivorship_bias_risk"
    missing = [
        column
        for column in ("delisting_date", "membership_start_date")
        if column not in universe.columns
    ]
    has_delisted = (
        "is_active" in universe.columns
        and not universe["is_active"].fillna(True).astype(bool).all()
    )
    if missing and not has_delisted:
        return AuditIssue(
            severity="warning",
            check_name=check,
            passed=False,
            affected_dates=[],
            affected_tickers=[],
            message="Universe appears to be a current constituent list without delisting history.",
            suggested_fix="Use point-in-time membership and include delisted/merged/suspended historical names.",
        )
    return passed_issue(check, "Universe contains fields that help audit survivorship bias.")


def check_same_frequency(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
) -> AuditIssue:
    """Detect strategy/benchmark return frequency mismatch from date spacing."""
    check = "return_frequency_mismatch"
    strategy_dates = _date_index(strategy_returns)
    benchmark_dates = _date_index(benchmark_returns)
    if len(strategy_dates) < 3 or len(benchmark_dates) < 3:
        return passed_issue(check, "Not enough observations to infer return frequency.")

    strategy_gap = strategy_dates.to_series().diff().dropna().median()
    benchmark_gap = benchmark_dates.to_series().diff().dropna().median()
    if strategy_gap != benchmark_gap:
        return AuditIssue(
            severity="warning",
            check_name=check,
            passed=False,
            affected_dates=[],
            affected_tickers=[],
            message=f"Strategy median spacing {strategy_gap} differs from benchmark {benchmark_gap}.",
            suggested_fix="Compare returns sampled at the same daily/monthly frequency.",
        )
    return passed_issue(check, "Strategy and benchmark return frequencies appear aligned.")


def _failed_schema(check_name: str, message: str) -> AuditIssue:
    return AuditIssue(
        severity="critical",
        check_name=check_name,
        passed=False,
        affected_dates=[],
        affected_tickers=[],
        message=message,
        suggested_fix="Provide the required columns before running this audit check.",
    )


def _normalize_ticker_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    result = df.copy()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result[date_column] = pd.to_datetime(result[date_column])
    return result


def _issue_from_bad_rows(
    check_name: str,
    bad: pd.DataFrame,
    severity: Severity,
    message: str,
    suggested_fix: str,
) -> AuditIssue:
    if bad.empty:
        return passed_issue(check_name, message.replace(" exist", " do not exist"))
    dates = []
    if "date" in bad.columns:
        dates = sorted({pd.Timestamp(value).date().isoformat() for value in bad["date"].dropna()})
    tickers = []
    if "ticker" in bad.columns:
        tickers = sorted({str(value).zfill(6) for value in bad["ticker"].dropna()})
    return AuditIssue(
        severity=severity,
        check_name=check_name,
        passed=False,
        affected_dates=dates,
        affected_tickers=tickers,
        message=message,
        suggested_fix=suggested_fix,
    )


def _date_index(data: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(data, pd.DataFrame) and "date" in data.columns:
        return pd.DatetimeIndex(pd.to_datetime(data["date"])).sort_values()
    return pd.DatetimeIndex(pd.to_datetime(data.index)).sort_values()


__all__ = [
    "AuditIssue",
    "Severity",
    "check_backfilled_missing_data",
    "check_fundamentals_available_date_usage",
    "check_future_universe_membership",
    "check_prices_before_listing",
    "check_same_day_signal_execution",
    "check_same_frequency",
    "check_survivorship_bias_risk",
]
