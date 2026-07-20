"""Return-series based performance analytics.

All metric functions consume returns, not prices. Returns are decimal period returns
(`0.01` means +1%). Annualized statistics use 252 trading days by default unless configured.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance calculation settings.

    Attributes:
        annualization_days: Trading periods per year. Unit: days/year. Default is 252.
        risk_free_rate: Annual risk-free rate as a decimal. Unit: annual return.
        min_observations: Minimum observations before warning that metrics may be unstable.
    """

    annualization_days: int = 252
    risk_free_rate: float = 0.0
    min_observations: int = 30

    def validate(self) -> None:
        """Validate annualization and risk-free settings."""
        if self.annualization_days < 1:
            raise ValueError("annualization_days must be positive")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")


def calculate_performance_metrics(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
    config: PerformanceConfig | None = None,
) -> dict[str, float | pd.Timestamp | None]:
    """Calculate core performance metrics from a return series.

    Formulas and units:
    - Total Return: compounded return, decimal.
    - CAGR: `(ending_value / starting_value) ** (1 / years) - 1`, decimal/year.
    - Annualized Volatility: daily return std * sqrt(annualization_days), decimal/year.
    - Sharpe Ratio: `(CAGR - risk_free_rate) / annualized_volatility`, unitless.
    - Sortino Ratio: `(CAGR - risk_free_rate) / annualized_downside_volatility`, unitless.
    - Maximum Drawdown: minimum wealth / prior peak - 1, decimal.
    - Calmar Ratio: `(CAGR - risk_free_rate) / abs(max_drawdown)`, unitless.
    - Turnover and transaction cost metrics are decimal fractions of portfolio value.
    - Benchmark excess/tracking/information/beta/alpha are annualized where applicable.
    """
    resolved = config or PerformanceConfig()
    resolved.validate()
    clean_returns = _clean_returns(returns)
    _warn_if_insufficient(clean_returns, resolved)

    wealth = (1.0 + clean_returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0) if not wealth.empty else 0.0
    years = max(len(clean_returns) / resolved.annualization_days, 1 / resolved.annualization_days)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if clean_returns.size else 0.0
    volatility = _annualized_volatility(clean_returns, resolved)
    downside = clean_returns.loc[clean_returns < 0.0]
    downside_volatility = _annualized_volatility(downside, resolved)
    sharpe = _safe_div(cagr - resolved.risk_free_rate, volatility)
    sortino = _safe_div(cagr - resolved.risk_free_rate, downside_volatility)
    drawdown = calculate_drawdown_series(clean_returns)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    drawdown_period = calculate_drawdown_period(clean_returns)
    monthly = calculate_monthly_return_table(clean_returns)
    monthly_returns = monthly["monthly_return"] if not monthly.empty else pd.Series(dtype="float64")

    benchmark_stats = _benchmark_metrics(clean_returns, benchmark_returns, resolved)
    avg_monthly_turnover = _average_monthly_turnover(turnover)
    annual_turnover = avg_monthly_turnover * 12.0
    total_transaction_cost = float(_clean_optional_series(transaction_cost).sum())

    metrics: dict[str, float | pd.Timestamp | None] = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": max_drawdown,
        "calmar_ratio": _safe_div(cagr - resolved.risk_free_rate, abs(max_drawdown)),
        "monthly_win_rate": float((monthly_returns > 0.0).mean()) if len(monthly_returns) else 0.0,
        "best_month": float(monthly_returns.max()) if len(monthly_returns) else 0.0,
        "worst_month": float(monthly_returns.min()) if len(monthly_returns) else 0.0,
        "average_monthly_turnover": avg_monthly_turnover,
        "annual_turnover": annual_turnover,
        "total_transaction_cost": total_transaction_cost,
        "drawdown_start_date": drawdown_period["start_date"],
        "drawdown_trough_date": drawdown_period["trough_date"],
        "drawdown_recovery_date": drawdown_period["recovery_date"],
    }
    metrics.update(benchmark_stats)
    return metrics


def metrics_to_frame(
    metrics: dict[str, object], *, strategy_name: str = "strategy"
) -> pd.DataFrame:
    """Return metrics as a two-column DataFrame with metric names and values."""
    return pd.DataFrame(
        {
            "strategy": strategy_name,
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
        }
    )


def compare_strategies(
    returns_by_strategy: dict[str, pd.Series],
    *,
    benchmark_returns: pd.Series | None = None,
    config: PerformanceConfig | None = None,
) -> pd.DataFrame:
    """Create a strategy comparison table with metrics as columns."""
    rows = []
    for name, returns in sorted(returns_by_strategy.items()):
        row = {"strategy": name}
        row.update(
            calculate_performance_metrics(
                returns,
                benchmark_returns=benchmark_returns,
                config=config,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_yearly_returns(returns: pd.Series) -> pd.DataFrame:
    """Return compounded calendar-year returns. Unit: decimal annual return."""
    clean = _clean_returns(returns)
    result = (
        clean.groupby(pd.Series(clean.index.year, index=clean.index, name="year"))
        .apply(lambda series: float((1.0 + series).prod() - 1.0))
        .reset_index(name="return")
    )
    return result


def calculate_monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Return compounded monthly returns. Unit: decimal monthly return."""
    clean = _clean_returns(returns)
    if clean.empty:
        return pd.DataFrame(columns=["year", "month", "monthly_return"])
    years = pd.Series(clean.index.year, index=clean.index, name="year")
    months = pd.Series(clean.index.month, index=clean.index, name="month")
    result = (
        clean.groupby([years, months])
        .apply(lambda series: float((1.0 + series).prod() - 1.0))
        .reset_index(name="monthly_return")
    )
    return result


def calculate_rolling_metrics(
    returns: pd.Series,
    *,
    window: int = 252,
    config: PerformanceConfig | None = None,
) -> pd.DataFrame:
    """Return rolling 12M return, Sharpe, and volatility.

    Units: rolling return is decimal per window; volatility is decimal/year; Sharpe is unitless.
    """
    resolved = config or PerformanceConfig()
    resolved.validate()
    clean = _clean_returns(returns)
    result = pd.DataFrame(index=clean.index)
    result["rolling_12m_return"] = (1.0 + clean).rolling(window).apply(np.prod, raw=True) - 1.0
    result["rolling_12m_volatility"] = clean.rolling(window).std(ddof=0) * math.sqrt(
        resolved.annualization_days
    )
    excess_daily = clean - resolved.risk_free_rate / resolved.annualization_days
    rolling_excess = excess_daily.rolling(window).mean() * resolved.annualization_days
    result["rolling_12m_sharpe"] = rolling_excess / result["rolling_12m_volatility"]
    return result.reset_index(names="date")


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """Return drawdown series. Unit: decimal below previous high-water mark."""
    clean = _clean_returns(returns)
    wealth = (1.0 + clean).cumprod()
    return wealth / wealth.cummax() - 1.0


def calculate_drawdown_period(returns: pd.Series) -> dict[str, pd.Timestamp | None]:
    """Return maximum drawdown start, trough, and recovery dates."""
    clean = _clean_returns(returns)
    if clean.empty:
        return {"start_date": None, "trough_date": None, "recovery_date": None}
    wealth = (1.0 + clean).cumprod()
    high_water = wealth.cummax()
    drawdown = wealth / high_water - 1.0
    trough_date = pd.Timestamp(drawdown.idxmin())
    start_candidates = wealth.loc[:trough_date]
    start_date = pd.Timestamp(start_candidates.idxmax())
    recovery = wealth.loc[trough_date:]
    recovered = recovery.loc[recovery >= float(high_water.loc[trough_date])]
    recovery_date = pd.Timestamp(recovered.index[0]) if not recovered.empty else None
    return {
        "start_date": start_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


def calculate_regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    config: PerformanceConfig | None = None,
) -> pd.DataFrame:
    """Return performance metrics by market regime."""
    clean = _clean_returns(returns)
    regime_series = pd.Series(regimes).reindex(clean.index)
    rows = []
    for regime, idx in regime_series.groupby(regime_series).groups.items():
        row = {"regime": regime}
        row.update(calculate_performance_metrics(clean.loc[idx], config=config))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def calculate_group_contribution(
    contributions: pd.DataFrame,
    *,
    group_column: str,
    contribution_column: str = "contribution",
) -> pd.DataFrame:
    """Aggregate sector, stock, or factor contribution. Unit: decimal return contribution."""
    required = {group_column, contribution_column}
    missing = required - set(contributions.columns)
    if missing:
        raise ValueError(f"Missing contribution columns: {', '.join(sorted(missing))}")
    result = (
        contributions.groupby(group_column, sort=True)[contribution_column]
        .sum()
        .reset_index(name="total_contribution")
    )
    total = float(result["total_contribution"].sum())
    result["contribution_share"] = result["total_contribution"] / total if total != 0.0 else 0.0
    return result


def export_metrics(
    metrics: dict[str, object],
    output_dir: str | Path,
    *,
    strategy_name: str = "strategy",
) -> None:
    """Export metrics to CSV and JSON files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics_to_frame(metrics, strategy_name=strategy_name).to_csv(path / "metrics.csv", index=False)
    serializable = {key: _json_value(value) for key, value in metrics.items()}
    (path / "metrics.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def calculate_daily_results(values: pd.DataFrame) -> pd.DataFrame:
    """Add daily return and drawdown columns to a portfolio value series."""
    required = {"date", "portfolio_value"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"Missing value columns: {', '.join(sorted(missing))}")

    result = values.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)
    result["daily_return"] = result["portfolio_value"].pct_change().fillna(0.0)
    high_water = result["portfolio_value"].cummax()
    result["drawdown"] = result["portfolio_value"] / high_water - 1.0
    return result


def calculate_monthly_results(daily_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily results into month-end return rows."""
    if daily_results.empty:
        return pd.DataFrame(
            columns=["month", "month_end_date", "monthly_return", "portfolio_value"]
        )

    result = daily_results.copy()
    result["month"] = pd.to_datetime(result["date"]).dt.to_period("M").astype(str)
    grouped = result.groupby("month", sort=True)
    monthly = grouped.agg(
        month_end_date=("date", "last"),
        start_value=("portfolio_value", "first"),
        end_value=("portfolio_value", "last"),
    ).reset_index()
    monthly["monthly_return"] = monthly["end_value"] / monthly["start_value"] - 1.0
    monthly = monthly.rename(columns={"end_value": "portfolio_value"})
    return monthly[["month", "month_end_date", "monthly_return", "portfolio_value"]]


def calculate_performance_summary(
    daily_results: pd.DataFrame,
    *,
    annualization_days: int = 252,
) -> dict[str, float]:
    """Return engine-compatible summary metrics from daily result rows."""
    if daily_results.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    returns = pd.Series(
        daily_results["daily_return"].to_numpy(),
        index=pd.to_datetime(daily_results["date"]),
    )
    metrics = calculate_performance_metrics(
        returns,
        turnover=daily_results.get("turnover"),
        transaction_cost=daily_results.get("transaction_cost"),
        config=PerformanceConfig(annualization_days=annualization_days, min_observations=1),
    )
    return {
        "total_return": float(metrics["total_return"]),
        "annualized_return": float(metrics["cagr"]),
        "annualized_volatility": float(metrics["annualized_volatility"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "max_drawdown": float(metrics["maximum_drawdown"]),
    }


def _benchmark_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    config: PerformanceConfig,
) -> dict[str, float]:
    if benchmark_returns is None:
        return {
            "benchmark_excess_return": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
        }
    benchmark = _clean_returns(benchmark_returns).reindex(returns.index).fillna(0.0)
    excess = returns - benchmark
    strategy_total = float((1.0 + returns).prod() - 1.0)
    benchmark_total = float((1.0 + benchmark).prod() - 1.0)
    tracking_error = float(excess.std(ddof=0) * math.sqrt(config.annualization_days))
    information_ratio = _safe_div(excess.mean() * config.annualization_days, tracking_error)
    benchmark_var = float(benchmark.var(ddof=0))
    beta = float(returns.cov(benchmark, ddof=0) / benchmark_var) if benchmark_var > 0.0 else 0.0
    strategy_cagr = (1.0 + strategy_total) ** (
        config.annualization_days / max(len(returns), 1)
    ) - 1.0
    benchmark_cagr = (1.0 + benchmark_total) ** (
        config.annualization_days / max(len(benchmark), 1)
    ) - 1.0
    alpha = float(
        (strategy_cagr - config.risk_free_rate) - beta * (benchmark_cagr - config.risk_free_rate)
    )
    return {
        "benchmark_excess_return": strategy_total - benchmark_total,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
    }


def _average_monthly_turnover(turnover: pd.Series | None) -> float:
    clean = _clean_optional_series(turnover)
    if clean.empty:
        return 0.0
    if isinstance(clean.index, pd.DatetimeIndex):
        return float(clean.groupby([clean.index.year, clean.index.month]).sum().mean())
    return float(clean.mean())


def _clean_returns(returns: pd.Series) -> pd.Series:
    clean = pd.Series(returns).copy()
    clean.index = pd.to_datetime(clean.index)
    clean = pd.to_numeric(clean, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return clean.astype(float).sort_index()


def _clean_optional_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    clean = pd.Series(series).copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.RangeIndex(len(clean))
    else:
        clean.index = pd.to_datetime(clean.index)
    return (
        pd.to_numeric(clean, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )


def _annualized_volatility(returns: pd.Series, config: PerformanceConfig) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(config.annualization_days))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator)


def _warn_if_insufficient(returns: pd.Series, config: PerformanceConfig) -> None:
    if len(returns) < config.min_observations:
        warnings.warn(
            f"Only {len(returns)} observations available; performance metrics may be unstable.",
            RuntimeWarning,
            stacklevel=2,
        )


def _json_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "PerformanceConfig",
    "calculate_daily_results",
    "calculate_drawdown_period",
    "calculate_drawdown_series",
    "calculate_group_contribution",
    "calculate_monthly_results",
    "calculate_monthly_return_table",
    "calculate_performance_metrics",
    "calculate_performance_summary",
    "calculate_regime_performance",
    "calculate_rolling_metrics",
    "calculate_yearly_returns",
    "compare_strategies",
    "export_metrics",
    "metrics_to_frame",
]
