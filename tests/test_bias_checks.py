"""Tests for backtest bias and data-quality audit checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.validation.backtest_audit import (
    AuditConfig,
    BacktestAuditError,
    run_backtest_audit,
    validate_or_raise,
)
from src.validation.bias_checks import (
    check_fundamentals_available_date_usage,
    check_prices_before_listing,
    check_same_day_signal_execution,
)
from src.validation.data_quality import (
    check_benchmark_date_alignment,
    check_duplicate_prices,
    check_portfolio_weight_sums,
    check_transaction_costs_reflected,
)


def test_detects_prices_before_listing() -> None:
    issue = check_prices_before_listing(_prices(), _universe())

    assert not issue.passed
    assert issue.severity == "critical"
    assert issue.affected_tickers == ["000001"]
    assert "2023-12-29" in issue.affected_dates


def test_detects_available_date_lookahead() -> None:
    factors = pd.DataFrame(
        {
            "calculation_date": [pd.Timestamp("2024-01-31")],
            "available_date": [pd.Timestamp("2024-02-15")],
            "ticker": ["000001"],
        }
    )

    issue = check_fundamentals_available_date_usage(factors)

    assert not issue.passed
    assert issue.check_name == "fundamentals_available_date_usage"


def test_detects_same_day_signal_execution() -> None:
    trades = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-31")],
            "execution_date": [pd.Timestamp("2024-01-31")],
            "ticker": ["000001"],
        }
    )

    issue = check_same_day_signal_execution(trades)

    assert not issue.passed
    assert issue.severity == "critical"


def test_detects_duplicate_prices_weight_errors_and_missing_costs() -> None:
    duplicate_issue = check_duplicate_prices(
        pd.concat([_prices(), _prices().iloc[[0]]], ignore_index=True)
    )
    weight_issue = check_portfolio_weight_sums(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31")],
                "ticker": ["000001", "CASH"],
                "target_weight": [0.6, 0.3],
            }
        )
    )
    cost_issue = check_transaction_costs_reflected(
        pd.DataFrame({"ticker": ["000001"], "trade_value": [1000.0], "transaction_cost": [0.0]})
    )

    assert not duplicate_issue.passed
    assert not weight_issue.passed
    assert not cost_issue.passed


def test_detects_benchmark_date_mismatch() -> None:
    strategy = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    benchmark = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))

    issue = check_benchmark_date_alignment(strategy, benchmark)

    assert not issue.passed
    assert set(issue.affected_dates) == {"2024-01-03", "2024-01-04"}


def test_audit_report_exports_and_strict_mode_blocks(tmp_path: Path) -> None:
    report = run_backtest_audit(
        prices=_prices(),
        universe=_universe(),
        factor_scores=pd.DataFrame(
            {
                "calculation_date": [pd.Timestamp("2024-01-31")],
                "available_date": [pd.Timestamp("2024-02-15")],
                "ticker": ["000001"],
            }
        ),
        trades=pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-31")],
                "execution_date": [pd.Timestamp("2024-02-01")],
                "ticker": ["000001"],
                "trade_value": [1000.0],
                "transaction_cost": [2.0],
            }
        ),
        output_dir=tmp_path,
        config=AuditConfig(mode="warning", halt_on_failure=False),
    )

    assert {
        "severity",
        "check_name",
        "passed",
        "affected_dates",
        "affected_tickers",
        "message",
        "suggested_fix",
    } <= set(report.columns)
    assert (tmp_path / "backtest_audit.csv").exists()
    assert (tmp_path / "backtest_audit.json").exists()
    with pytest.raises(BacktestAuditError):
        validate_or_raise(report, mode="warning")


def test_warning_mode_allows_warnings_but_strict_blocks() -> None:
    report = run_backtest_audit(
        universe=pd.DataFrame({"ticker": ["000001"], "listing_date": [pd.Timestamp("2020-01-01")]}),
        config=AuditConfig(mode="warning", halt_on_failure=False),
    )

    validate_or_raise(report, mode="warning")
    with pytest.raises(BacktestAuditError):
        validate_or_raise(report, mode="strict")


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-02")],
            "ticker": ["000001", "000001"],
            "adjusted_close": [100.0, 101.0],
            "is_suspended": [False, False],
        }
    )


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["000001"],
            "listing_date": [pd.Timestamp("2024-01-02")],
        }
    )
