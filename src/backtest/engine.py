"""Look-ahead-safe backtest engine for MUST30 strategies."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.backtest.costs import TransactionCostConfig
from src.backtest.metrics import (
    calculate_daily_results,
    calculate_monthly_results,
    calculate_performance_summary,
)
from src.database.models import BacktestDaily
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.optimizer import WeightConstraints
from src.portfolio.rebalance import RebalanceConfig, execute_monthly_rebalance
from src.portfolio.selector import select_monthly_portfolio
from src.regime.market_regime import classify_market_regime

LOGGER = logging.getLogger(__name__)

StrategyName = Literal["score_weight", "equal_weight", "rank_weight"]
RebalanceFrequency = Literal["monthly", "quarterly"]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a deterministic point-in-time backtest."""

    start_date: str = "2014-01-01"
    end_date: str | None = None
    strategy: StrategyName = "score_weight"
    initial_capital: float = 1_000_000_000.0
    transaction_cost: float = 0.002
    commission: float = 0.0015
    market_impact: float = 0.0005
    execution_price: Literal["next_open", "next_close"] = "next_open"
    trade_band: float = 0.01
    target_size: int = 30
    rebalance_frequency: RebalanceFrequency = "monthly"
    regime_equity_weights: dict[str, float] = field(
        default_factory=lambda: {"Risk-On": 1.0, "Neutral": 0.8, "Risk-Off": 0.5}
    )
    factor_setting: str = "balanced"
    strategy_name: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output tables and metadata."""

    daily_results: pd.DataFrame
    monthly_results: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    factor_scores: pd.DataFrame
    regime_history: pd.DataFrame
    performance_summary: dict[str, float | str]
    metadata: dict[str, object]


def run_backtest(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    factor_scores: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    output_dir: str | Path | None = None,
    db_session: Session | None = None,
    delisted_tickers: set[str] | None = None,
) -> BacktestResult:
    """Run a look-ahead-safe monthly MUST30 backtest.

    Dividend and corporate-action handling: portfolio valuation uses `adjusted_close` when
    available, so dividends and splits are assumed to be embedded in the adjusted price series.
    If `adjusted_close` is absent the engine falls back to `close` and does not model cash
    dividends separately.
    """
    resolved = config or BacktestConfig()
    start_date = pd.Timestamp(resolved.start_date).normalize()
    normalized_prices = _normalize_prices(prices)
    end_date = (
        pd.Timestamp(resolved.end_date).normalize()
        if resolved.end_date is not None
        else pd.Timestamp(normalized_prices["date"].max()).normalize()
    )
    normalized_prices = normalized_prices.loc[
        (normalized_prices["date"] >= start_date) & (normalized_prices["date"] <= end_date)
    ].copy()
    if normalized_prices.empty:
        raise ValueError("No prices available in backtest period")

    normalized_universe = _normalize_universe(universe)
    normalized_factors = _normalize_factors(factor_scores)
    regime_history = classify_market_regime(market_data)
    trading_dates = [
        pd.Timestamp(value).normalize() for value in sorted(normalized_prices["date"].unique())
    ]
    signal_dates = _signal_dates(trading_dates, resolved.rebalance_frequency)
    execution_by_signal = _execution_map(signal_dates, trading_dates)

    positions = pd.DataFrame(columns=["ticker", "shares"])
    cash_balance = float(resolved.initial_capital)
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []
    selected_factor_frames: list[pd.DataFrame] = []
    last_prices: dict[str, float] = {}
    latest_weights = pd.DataFrame()

    execution_to_signal = {execution: signal for signal, execution in execution_by_signal.items()}
    for current_date in trading_dates:
        try:
            day_prices = normalized_prices.loc[normalized_prices["date"] == current_date].copy()
            tradable_prices = day_prices.loc[~day_prices["is_suspended"].fillna(False).astype(bool)]
            last_prices.update(
                dict(
                    zip(tradable_prices["ticker"], tradable_prices["adjusted_close"], strict=False)
                )
            )

            if current_date in execution_to_signal:
                signal_date = execution_to_signal[current_date]
                target_weights, selected = _target_weights_for_signal(
                    normalized_factors,
                    normalized_universe,
                    normalized_prices,
                    regime_history,
                    signal_date,
                    current_date,
                    resolved,
                )
                selected_factor_frames.append(selected.assign(signal_date=signal_date))
                unavailable = _unavailable_tickers(day_prices, delisted_tickers or set())
                rebalance = execute_monthly_rebalance(
                    positions,
                    target_weights,
                    normalized_prices,
                    signal_date,
                    cash_balance=cash_balance,
                    config=RebalanceConfig(
                        trade_band=resolved.trade_band,
                        execution_price=resolved.execution_price,
                        cost_config=_transaction_cost_config(resolved),
                    ),
                    unavailable_tickers=unavailable,
                )
                positions = _positions_after_trades(rebalance.trades)
                cash_balance = rebalance.cash_balance
                latest_weights = rebalance.post_trade_weights.assign(date=current_date)
                trade_frames.append(rebalance.trades)
                turnover = rebalance.turnover
                transaction_cost = rebalance.transaction_cost
            else:
                turnover = 0.0
                transaction_cost = 0.0

            portfolio_value = _portfolio_value(positions, cash_balance, last_prices)
            cash_weight = cash_balance / portfolio_value if portfolio_value > 0.0 else 0.0
            daily_rows.append(
                {
                    "date": current_date,
                    "strategy_name": _strategy_name(resolved),
                    "portfolio_value": portfolio_value,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "cash_weight": cash_weight,
                }
            )
            holding_rows.extend(
                _holding_rows(current_date, positions, cash_balance, last_prices, portfolio_value)
            )
        except Exception:
            LOGGER.exception("Backtest failed on date=%s", current_date.date())
            raise

    daily_results = calculate_daily_results(pd.DataFrame(daily_rows))
    monthly_results = calculate_monthly_results(daily_results)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    holdings = pd.DataFrame(holding_rows)
    selected_factors = (
        pd.concat(selected_factor_frames, ignore_index=True)
        if selected_factor_frames
        else pd.DataFrame()
    )
    summary = calculate_performance_summary(daily_results)
    summary["strategy_name"] = _strategy_name(resolved)
    metadata = _metadata(resolved, start_date, end_date)
    result = BacktestResult(
        daily_results=daily_results,
        monthly_results=monthly_results,
        holdings=holdings,
        trades=trades,
        factor_scores=selected_factors,
        regime_history=regime_history,
        performance_summary=summary,
        metadata=metadata,
    )
    if output_dir is not None:
        save_backtest_to_parquet(result, output_dir)
    if db_session is not None:
        save_backtest_to_db(result, db_session)
    del latest_weights
    return result


def save_backtest_to_parquet(result: BacktestResult, output_dir: str | Path) -> None:
    """Save all backtest outputs and metadata to a directory of parquet/json files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.daily_results.to_parquet(path / "daily_results.parquet", index=False)
    result.monthly_results.to_parquet(path / "monthly_results.parquet", index=False)
    result.holdings.to_parquet(path / "holdings.parquet", index=False)
    result.trades.to_parquet(path / "trades.parquet", index=False)
    result.factor_scores.to_parquet(path / "factor_scores.parquet", index=False)
    result.regime_history.to_parquet(path / "regime_history.parquet", index=False)
    (path / "performance_summary.json").write_text(
        json.dumps(result.performance_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_backtest_to_db(result: BacktestResult, session: Session) -> None:
    """Persist daily backtest results to the existing backtest_daily table."""
    strategy_name = str(result.performance_summary["strategy_name"])
    session.execute(delete(BacktestDaily).where(BacktestDaily.strategy_name == strategy_name))
    for row in result.daily_results.itertuples(index=False):
        session.add(
            BacktestDaily(
                date=pd.Timestamp(row.date).date(),
                strategy_name=strategy_name,
                daily_return=float(row.daily_return),
                portfolio_value=float(row.portfolio_value),
                benchmark_return=None,
                benchmark_value=None,
                drawdown=float(row.drawdown),
                turnover=float(row.turnover),
                transaction_cost=float(row.transaction_cost),
                cash_weight=float(row.cash_weight),
            )
        )
    session.flush()


def _target_weights_for_signal(
    factor_scores: pd.DataFrame,
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    regime_history: pd.DataFrame,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del execution_date
    pit_universe = universe.loc[
        (universe["listing_date"] <= signal_date)
        & ((universe["delisting_date"].isna()) | (universe["delisting_date"] > signal_date))
    ].copy()
    factor_mask = factor_scores["calculation_date"] <= signal_date
    if "available_date" in factor_scores.columns:
        factor_mask &= factor_scores["available_date"] <= signal_date
    pit_factors = factor_scores.loc[factor_mask].copy()
    selection_constraints = PortfolioConstraints(
        target_size=config.target_size,
        max_sector_count=config.target_size,
        max_kosdaq_count=config.target_size,
        min_core_count=0,
        min_defensive_count=0,
        min_listing_trading_days=1,
        min_avg_trading_value_20d=0.0,
        min_available_factors=1,
        relaxation_order=(),
    )
    selected = select_monthly_portfolio(
        pit_factors,
        pit_universe,
        signal_date,
        constraints=selection_constraints,
        price_history=prices.loc[prices["date"] <= signal_date],
    ).selected_portfolio
    if len(selected) != config.target_size:
        raise ValueError(f"Selection failed on {signal_date.date()}: got {len(selected)} stocks")

    regime = _regime_on_date(regime_history, signal_date)
    from src.portfolio.rebalance import calculate_rebalance_weights

    target = calculate_rebalance_weights(
        selected,
        signal_date,
        regime=regime,
        method=config.strategy,
        equity_weight=float(config.regime_equity_weights.get(regime, 0.8)),
        constraints=WeightConstraints(
            target_size=config.target_size,
            max_stock_weight=max(0.07, 1.0 / config.target_size),
            max_sector_weight=1.0,
            max_kosdaq_weight=1.0,
        ),
    )
    return target, selected


def _regime_on_date(regime_history: pd.DataFrame, signal_date: pd.Timestamp) -> str:
    history = regime_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    available = history.loc[history["date"] <= signal_date]
    if available.empty:
        return "Neutral"
    return str(available.sort_values("date").iloc[-1]["regime"])


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    if "adjusted_close" not in result.columns:
        result["adjusted_close"] = result["close"]
    if "is_suspended" not in result.columns:
        result["is_suspended"] = False
    for column in ("open", "close", "adjusted_close"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def _normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "market", "sector", "universe_role", "listing_date"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Missing universe columns: {', '.join(sorted(missing))}")
    result = universe.copy()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["listing_date"] = pd.to_datetime(result["listing_date"]).dt.normalize()
    if "data_start_date" not in result.columns:
        result["data_start_date"] = result["listing_date"]
    result["data_start_date"] = pd.to_datetime(result["data_start_date"]).dt.normalize()
    if "delisting_date" in result.columns:
        result["delisting_date"] = pd.to_datetime(result["delisting_date"]).dt.normalize()
    else:
        result["delisting_date"] = pd.NaT
    if "is_active" not in result.columns:
        result["is_active"] = True
    return result


def _normalize_factors(factor_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"calculation_date", "ticker", "composite_score"}
    missing = required - set(factor_scores.columns)
    if missing:
        raise ValueError(f"Missing factor score columns: {', '.join(sorted(missing))}")
    result = factor_scores.copy()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["calculation_date"] = pd.to_datetime(result["calculation_date"]).dt.normalize()
    if "available_date" in result.columns:
        result["available_date"] = pd.to_datetime(result["available_date"]).dt.normalize()
    return result.sort_values(["calculation_date", "ticker"])


def _signal_dates(
    trading_dates: list[pd.Timestamp],
    rebalance_frequency: RebalanceFrequency = "monthly",
) -> list[pd.Timestamp]:
    dates = pd.Series(trading_dates)
    month_end = dates.loc[dates.dt.to_period("M").ne(dates.dt.to_period("M").shift(-1))]
    if rebalance_frequency == "monthly":
        selected = month_end
    elif rebalance_frequency == "quarterly":
        selected = month_end.loc[month_end.dt.month.isin([3, 6, 9, 12])]
    else:
        raise ValueError("rebalance_frequency must be monthly or quarterly")
    return [pd.Timestamp(value).normalize() for value in selected.tolist()]


def _execution_map(
    signal_dates: list[pd.Timestamp],
    trading_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Timestamp]:
    result: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal in signal_dates:
        future = [date for date in trading_dates if date > signal]
        if future:
            result[signal] = future[0]
    return result


def _unavailable_tickers(day_prices: pd.DataFrame, delisted_tickers: set[str]) -> set[str]:
    suspended = set(
        day_prices.loc[day_prices["is_suspended"].fillna(False).astype(bool), "ticker"].astype(str)
    )
    return suspended | {ticker.zfill(6) for ticker in delisted_tickers}


def _positions_after_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "shares"])
    result = trades.copy()
    result["shares"] = result["shares_before"] + result["trade_shares"]
    return result.loc[result["shares"].abs() > 1e-10, ["ticker", "shares"]].reset_index(drop=True)


def _portfolio_value(
    positions: pd.DataFrame,
    cash_balance: float,
    last_prices: dict[str, float],
) -> float:
    stock_value = 0.0
    for row in positions.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker not in last_prices:
            raise ValueError(f"Missing valuation price for ticker={ticker}")
        stock_value += float(row.shares) * float(last_prices[ticker])
    return stock_value + float(cash_balance)


def _holding_rows(
    date: pd.Timestamp,
    positions: pd.DataFrame,
    cash_balance: float,
    last_prices: dict[str, float],
    portfolio_value: float,
) -> list[dict[str, object]]:
    rows = []
    for row in positions.itertuples(index=False):
        ticker = str(row.ticker)
        value = float(row.shares) * float(last_prices[ticker])
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "shares": float(row.shares),
                "market_value": value,
                "weight": value / portfolio_value,
            }
        )
    rows.append(
        {
            "date": date,
            "ticker": "CASH",
            "shares": 0.0,
            "market_value": cash_balance,
            "weight": cash_balance / portfolio_value,
        }
    )
    return rows


def _strategy_name(config: BacktestConfig) -> str:
    return config.strategy_name or f"MUST30 {config.strategy}"


def _transaction_cost_config(config: BacktestConfig) -> TransactionCostConfig:
    """Return the one-way commission and market-impact assumptions for a backtest."""
    if config.commission < 0.0 or config.market_impact < 0.0:
        raise ValueError("commission and market_impact must be non-negative")
    configured_total = config.commission + config.market_impact
    if abs(configured_total - config.transaction_cost) > 1e-12:
        LOGGER.warning(
            "Backtest transaction_cost %.6f differs from commission + market_impact %.6f; "
            "using explicit commission and market_impact rates.",
            config.transaction_cost,
            configured_total,
        )
    return TransactionCostConfig(
        commission_rate=config.commission,
        market_impact_rate=config.market_impact,
    )


def _metadata(
    config: BacktestConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, object]:
    values = asdict(config)
    values["start_date"] = str(start_date.date())
    values["end_date"] = str(end_date.date())
    values["data_policy"] = (
        "Point-in-time universe and available_date factor filters; signals use month-end data "
        "and execute on the next trading day."
    )
    values["price_policy"] = (
        "Valuation uses adjusted_close when available; dividends/splits are assumed embedded in "
        "adjusted prices. close is used only as fallback."
    )
    return values


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "RebalanceFrequency",
    "run_backtest",
    "save_backtest_to_db",
    "save_backtest_to_parquet",
]
