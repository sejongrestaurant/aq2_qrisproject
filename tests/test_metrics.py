"""Tests for return-series performance metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.metrics import (
    PerformanceConfig,
    calculate_drawdown_period,
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
from src.backtest.report import build_performance_report, build_strategy_comparison


def test_core_metrics_match_manual_calculation() -> None:
    """Core metrics should match hand calculations on a simple return array."""
    returns = _returns([0.10, -0.05, 0.02, -0.01])
    benchmark = _returns([0.05, -0.02, 0.01, 0.00])
    config = PerformanceConfig(annualization_days=4, risk_free_rate=0.01, min_observations=1)

    metrics = calculate_performance_metrics(returns, benchmark_returns=benchmark, config=config)
    wealth = 1.10 * 0.95 * 1.02 * 0.99
    total_return = wealth - 1.0
    cagr = wealth - 1.0
    volatility = returns.std(ddof=0) * math.sqrt(4)
    downside_vol = returns[returns < 0].std(ddof=0) * math.sqrt(4)

    assert metrics["total_return"] == pytest.approx(total_return)
    assert metrics["cagr"] == pytest.approx(cagr)
    assert metrics["annualized_volatility"] == pytest.approx(volatility)
    assert metrics["sharpe_ratio"] == pytest.approx((cagr - 0.01) / volatility)
    assert metrics["sortino_ratio"] == pytest.approx((cagr - 0.01) / downside_vol)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.05)
    assert metrics["calmar_ratio"] == pytest.approx((cagr - 0.01) / 0.05)


def test_monthly_metrics_and_turnover_costs() -> None:
    """Monthly win/best/worst and turnover metrics should use return series inputs."""
    returns = pd.Series(
        [0.10, -0.05, 0.02, -0.01],
        index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]),
    )
    turnover = pd.Series([0.1, 0.2, 0.3, 0.4], index=returns.index)
    costs = pd.Series([0.001, 0.002, 0.003, 0.004], index=returns.index)

    metrics = calculate_performance_metrics(
        returns,
        turnover=turnover,
        transaction_cost=costs,
        config=PerformanceConfig(annualization_days=4, min_observations=1),
    )

    assert metrics["monthly_win_rate"] == pytest.approx(0.5)
    assert metrics["best_month"] == pytest.approx(0.10)
    assert metrics["worst_month"] == pytest.approx(-0.05)
    assert metrics["average_monthly_turnover"] == pytest.approx(0.25)
    assert metrics["annual_turnover"] == pytest.approx(3.0)
    assert metrics["total_transaction_cost"] == pytest.approx(0.01)


def test_benchmark_relative_metrics_match_manual_calculation() -> None:
    """Tracking error, IR, beta, and alpha should match manual formulas."""
    returns = _returns([0.10, -0.05, 0.02, -0.01])
    benchmark = _returns([0.05, -0.02, 0.01, 0.00])
    config = PerformanceConfig(annualization_days=4, risk_free_rate=0.0, min_observations=1)

    metrics = calculate_performance_metrics(returns, benchmark_returns=benchmark, config=config)
    excess = returns - benchmark
    tracking_error = excess.std(ddof=0) * math.sqrt(4)
    beta = returns.cov(benchmark, ddof=0) / benchmark.var(ddof=0)

    assert metrics["benchmark_excess_return"] == pytest.approx(
        (1 + returns).prod() - 1 - ((1 + benchmark).prod() - 1)
    )
    assert metrics["tracking_error"] == pytest.approx(tracking_error)
    assert metrics["information_ratio"] == pytest.approx(excess.mean() * 4 / tracking_error)
    assert metrics["beta"] == pytest.approx(beta)


def test_nan_inf_are_removed_and_insufficient_observations_warns() -> None:
    """NaN and infinite values should be ignored, with a warning on short samples."""
    returns = pd.Series(
        [0.01, float("nan"), float("inf"), -float("inf"), 0.02],
        index=pd.bdate_range("2024-01-01", periods=5),
    )

    with pytest.warns(RuntimeWarning):
        metrics = calculate_performance_metrics(
            returns, config=PerformanceConfig(min_observations=3)
        )

    assert metrics["total_return"] == pytest.approx((1.01 * 1.02) - 1.0)


def test_drawdown_period_returns_start_trough_and_recovery() -> None:
    """MDD period should include start, trough, and recovery dates."""
    returns = _returns([0.10, -0.20, -0.10, 0.40, 0.01])

    period = calculate_drawdown_period(returns)

    assert period["start_date"] == returns.index[0]
    assert period["trough_date"] == returns.index[2]
    assert period["recovery_date"] == returns.index[3]


def test_tables_rolling_regime_and_contributions() -> None:
    """Additional analysis helpers should return DataFrames."""
    returns = _returns([0.01] * 260)
    yearly = calculate_yearly_returns(returns)
    monthly = calculate_monthly_return_table(returns)
    rolling = calculate_rolling_metrics(
        returns,
        window=3,
        config=PerformanceConfig(annualization_days=252, min_observations=1),
    )
    regimes = pd.Series(["Risk-On"] * 130 + ["Risk-Off"] * 130, index=returns.index)
    regime_perf = calculate_regime_performance(
        returns,
        regimes,
        config=PerformanceConfig(min_observations=1),
    )
    contribution = calculate_group_contribution(
        pd.DataFrame({"sector": ["A", "A", "B"], "contribution": [0.01, 0.02, -0.01]}),
        group_column="sector",
    )

    assert not yearly.empty
    assert not monthly.empty
    assert {"rolling_12m_return", "rolling_12m_sharpe", "rolling_12m_volatility"} <= set(
        rolling.columns
    )
    assert set(regime_perf["regime"]) == {"Risk-On", "Risk-Off"}
    assert contribution.loc[contribution["sector"] == "A", "total_contribution"].iloc[
        0
    ] == pytest.approx(0.03)


def test_metrics_frames_comparison_and_exports(tmp_path: Path) -> None:
    """Metrics should be convertible to DataFrame, comparable, and exportable."""
    returns = _returns([0.01, 0.02, -0.01, 0.03])
    metrics = calculate_performance_metrics(
        returns,
        config=PerformanceConfig(annualization_days=4, min_observations=1),
    )
    frame = metrics_to_frame(metrics, strategy_name="A")
    comparison = compare_strategies(
        {"B": returns * 0.5, "A": returns},
        config=PerformanceConfig(annualization_days=4, min_observations=1),
    )

    export_metrics(metrics, tmp_path, strategy_name="A")

    assert {"metric", "value", "strategy"} <= set(frame.columns)
    assert comparison["strategy"].tolist() == ["A", "B"]
    assert (tmp_path / "metrics.csv").exists()
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "total_return" in data


def test_report_builders_return_expected_sections() -> None:
    """Report helpers should assemble summary and comparison tables."""
    returns = _returns([0.01, 0.02, -0.01, 0.03])

    report = build_performance_report(
        returns,
        sector_contributions=pd.DataFrame({"sector": ["A"], "contribution": [0.01]}),
        stock_contributions=pd.DataFrame({"ticker": ["000001"], "contribution": [0.02]}),
        factor_contributions=pd.DataFrame({"factor": ["momentum"], "contribution": [0.03]}),
        config=PerformanceConfig(annualization_days=4, min_observations=1),
    )
    comparison = build_strategy_comparison(
        {"strategy": returns},
        config=PerformanceConfig(annualization_days=4, min_observations=1),
    )

    assert {"summary", "yearly_returns", "monthly_returns", "rolling_metrics"} <= set(report)
    assert "sector_contribution" in report
    assert comparison.loc[0, "strategy"] == "strategy"


def _returns(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)))
