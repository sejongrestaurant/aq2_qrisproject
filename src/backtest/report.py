"""Reporting helpers for backtest performance analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.metrics import (
    PerformanceConfig,
    calculate_group_contribution,
    calculate_monthly_return_table,
    calculate_performance_metrics,
    calculate_regime_performance,
    calculate_rolling_metrics,
    calculate_yearly_returns,
    compare_strategies,
    export_metrics,
    metrics_to_frame,
)


def build_performance_report(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    regimes: pd.Series | None = None,
    sector_contributions: pd.DataFrame | None = None,
    stock_contributions: pd.DataFrame | None = None,
    factor_contributions: pd.DataFrame | None = None,
    config: PerformanceConfig | None = None,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Build a dict of summary and analysis tables for a strategy."""
    metrics = calculate_performance_metrics(
        returns,
        benchmark_returns=benchmark_returns,
        config=config,
    )
    report: dict[str, pd.DataFrame | dict[str, object]] = {
        "summary": metrics,
        "summary_frame": metrics_to_frame(metrics),
        "yearly_returns": calculate_yearly_returns(returns),
        "monthly_returns": calculate_monthly_return_table(returns),
        "rolling_metrics": calculate_rolling_metrics(returns, config=config),
    }
    if regimes is not None:
        report["regime_performance"] = calculate_regime_performance(returns, regimes, config=config)
    if sector_contributions is not None:
        report["sector_contribution"] = calculate_group_contribution(
            sector_contributions,
            group_column="sector",
        )
    if stock_contributions is not None:
        report["stock_contribution"] = calculate_group_contribution(
            stock_contributions,
            group_column="ticker",
        )
    if factor_contributions is not None:
        report["factor_contribution"] = calculate_group_contribution(
            factor_contributions,
            group_column="factor",
        )
    return report


def build_strategy_comparison(
    returns_by_strategy: dict[str, pd.Series],
    *,
    benchmark_returns: pd.Series | None = None,
    config: PerformanceConfig | None = None,
) -> pd.DataFrame:
    """Return a strategy comparison table."""
    return compare_strategies(
        returns_by_strategy,
        benchmark_returns=benchmark_returns,
        config=config,
    )


def export_performance_report(
    report: dict[str, pd.DataFrame | dict[str, object]],
    output_dir: str | Path,
) -> None:
    """Export a report dict to CSV and JSON summary files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    if isinstance(summary, dict):
        export_metrics(summary, path)
    for name, value in report.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(path / f"{name}.csv", index=False)


__all__ = [
    "build_performance_report",
    "build_strategy_comparison",
    "export_performance_report",
]
