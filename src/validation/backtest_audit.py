"""Backtest audit orchestration and report export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.validation.bias_checks import (
    AuditIssue,
    check_backfilled_missing_data,
    check_fundamentals_available_date_usage,
    check_future_universe_membership,
    check_prices_before_listing,
    check_same_day_signal_execution,
    check_same_frequency,
    check_survivorship_bias_risk,
)
from src.validation.data_quality import (
    check_abnormal_price_jumps,
    check_benchmark_date_alignment,
    check_duplicate_prices,
    check_portfolio_weight_sums,
    check_suspended_abnormal_returns,
    check_transaction_costs_reflected,
)

AuditMode = Literal["strict", "warning"]


class BacktestAuditError(RuntimeError):
    """Raised when strict audit mode blocks a backtest run."""


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for pre-backtest audit behavior."""

    mode: AuditMode = "strict"
    halt_on_failure: bool = True
    abnormal_return_threshold: float = 0.30
    suspended_return_threshold: float = 0.02
    weight_tolerance: float = 1e-8

    def validate(self) -> None:
        """Validate audit mode and thresholds."""
        if self.mode not in {"strict", "warning"}:
            raise ValueError("mode must be strict or warning")
        if self.abnormal_return_threshold <= 0.0:
            raise ValueError("abnormal_return_threshold must be positive")
        if self.suspended_return_threshold < 0.0:
            raise ValueError("suspended_return_threshold must be non-negative")


def run_backtest_audit(
    *,
    prices: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    factor_scores: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    selections: pd.DataFrame | None = None,
    filled_data: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    daily_results: pd.DataFrame | None = None,
    strategy_returns: pd.Series | pd.DataFrame | None = None,
    benchmark_returns: pd.Series | pd.DataFrame | None = None,
    config: AuditConfig | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run available bias and data-quality checks and return an audit report DataFrame."""
    resolved = config or AuditConfig()
    resolved.validate()
    issues: list[AuditIssue] = []

    if prices is not None and universe is not None:
        issues.append(check_prices_before_listing(prices, universe))
    if factor_scores is not None:
        issues.append(check_fundamentals_available_date_usage(factor_scores))
    if trades is not None:
        issues.append(check_same_day_signal_execution(trades))
        issues.append(check_transaction_costs_reflected(trades, daily_results))
    if selections is not None and universe is not None:
        issues.append(check_future_universe_membership(selections, universe))
    if filled_data is not None:
        issues.append(check_backfilled_missing_data(filled_data))
    if universe is not None:
        issues.append(check_survivorship_bias_risk(universe))
    if prices is not None:
        issues.append(
            check_suspended_abnormal_returns(
                prices,
                return_threshold=resolved.suspended_return_threshold,
            )
        )
        issues.append(
            check_abnormal_price_jumps(
                prices,
                jump_threshold=resolved.abnormal_return_threshold,
            )
        )
        issues.append(check_duplicate_prices(prices))
    if weights is not None:
        issues.append(check_portfolio_weight_sums(weights, tolerance=resolved.weight_tolerance))
    if strategy_returns is not None and benchmark_returns is not None:
        issues.append(check_benchmark_date_alignment(strategy_returns, benchmark_returns))
        issues.append(check_same_frequency(strategy_returns, benchmark_returns))

    report = audit_issues_to_frame(issues)
    if output_dir is not None:
        export_audit_report(report, output_dir)
    if resolved.halt_on_failure:
        validate_or_raise(report, mode=resolved.mode)
    return report


def validate_or_raise(report: pd.DataFrame, *, mode: AuditMode = "strict") -> None:
    """Raise if the audit report fails under strict or warning mode."""
    if report.empty:
        return
    failed = report.loc[~report["passed"].astype(bool)]
    if mode == "strict":
        blocking = failed.loc[failed["severity"].isin(["critical", "warning"])]
    elif mode == "warning":
        blocking = failed.loc[failed["severity"] == "critical"]
    else:
        raise ValueError("mode must be strict or warning")
    if not blocking.empty:
        checks = ", ".join(blocking["check_name"].astype(str).tolist())
        raise BacktestAuditError(f"Backtest audit failed: {checks}")


def audit_issues_to_frame(issues: list[AuditIssue]) -> pd.DataFrame:
    """Convert audit issues to a normalized DataFrame."""
    rows = [issue.to_dict() for issue in issues]
    if not rows:
        return pd.DataFrame(
            columns=[
                "severity",
                "check_name",
                "passed",
                "affected_dates",
                "affected_tickers",
                "message",
                "suggested_fix",
            ]
        )
    result = pd.DataFrame(rows)
    result["affected_dates"] = result["affected_dates"].map(lambda values: ";".join(values))
    result["affected_tickers"] = result["affected_tickers"].map(lambda values: ";".join(values))
    return result.sort_values(["passed", "severity", "check_name"]).reset_index(drop=True)


def export_audit_report(report: pd.DataFrame, output_dir: str | Path) -> None:
    """Write audit report to CSV and JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report.to_csv(path / "backtest_audit.csv", index=False)
    records = report.to_dict(orient="records")
    (path / "backtest_audit.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "AuditConfig",
    "AuditIssue",
    "AuditMode",
    "BacktestAuditError",
    "audit_issues_to_frame",
    "export_audit_report",
    "run_backtest_audit",
    "validate_or_raise",
]
