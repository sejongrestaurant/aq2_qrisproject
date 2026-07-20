"""Rebalance helpers for target weights and trade execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.backtest.costs import TransactionCostConfig, calculate_transaction_cost
from src.portfolio.optimizer import WeightConstraints, WeightMethod, optimize_portfolio_weights
from src.regime.market_regime import get_equity_cash_allocation

ExecutionPrice = Literal["next_open", "next_close"]
TradabilityCallback = Callable[[str, pd.Timestamp], bool]


@dataclass(frozen=True)
class RebalanceConfig:
    """Configuration for monthly rebalance execution."""

    trade_band: float = 0.01
    execution_price: ExecutionPrice = "next_open"
    cost_config: TransactionCostConfig = TransactionCostConfig()

    def validate(self) -> None:
        """Validate rebalance settings."""
        if self.trade_band < 0.0:
            raise ValueError("trade_band must be non-negative")
        if self.execution_price not in {"next_open", "next_close"}:
            raise ValueError("execution_price must be next_open or next_close")
        self.cost_config.validate()


@dataclass(frozen=True)
class RebalanceResult:
    """Monthly rebalance output bundle."""

    trades: pd.DataFrame
    turnover: float
    transaction_cost: float
    post_trade_weights: pd.DataFrame
    cash_balance: float
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp


def calculate_rebalance_weights(
    selected_portfolio: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    *,
    regime: str = "Risk-On",
    method: WeightMethod = "score_weight",
    constraints: WeightConstraints | None = None,
    equity_weight: float | None = None,
) -> pd.DataFrame:
    """Return rebalance-date target weights including a CASH row."""
    resolved_equity = (
        get_equity_cash_allocation(regime)[0] if equity_weight is None else float(equity_weight)
    )
    weights = optimize_portfolio_weights(
        selected_portfolio,
        method=method,
        equity_weight=resolved_equity,
        constraints=constraints,
    )
    weights["rebalance_date"] = pd.Timestamp(rebalance_date)
    weights["regime"] = regime
    return weights


def execute_monthly_rebalance(
    current_positions: pd.DataFrame,
    target_weights: pd.DataFrame,
    price_data: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    cash_balance: float = 0.0,
    config: RebalanceConfig | None = None,
    unavailable_tickers: set[str] | None = None,
    tradability_callback: TradabilityCallback | None = None,
) -> RebalanceResult:
    """Execute a monthly rebalance on the first trading day after signal_date."""
    resolved = config or RebalanceConfig()
    resolved.validate()
    signal_ts = pd.Timestamp(signal_date).normalize()
    execution_ts = _next_execution_date(price_data, signal_ts)
    prices = _execution_prices(price_data, execution_ts, resolved.execution_price)
    positions = _normalize_positions(current_positions)
    targets = _normalize_targets(target_weights)

    tickers = sorted((set(positions["ticker"]) | set(targets["ticker"])) - {"CASH"})
    pre = _pre_trade_state(tickers, positions, prices, float(cash_balance))
    pre_value = float(pre["current_value"].sum()) + float(cash_balance)
    if pre_value <= 0.0:
        raise ValueError("pre-trade portfolio value must be positive")

    target_lookup = dict(zip(targets["ticker"], targets["target_weight"], strict=False))
    unavailable = {ticker.zfill(6) for ticker in (unavailable_tickers or set())}
    trade_rows: list[dict[str, object]] = []
    post_values = dict(zip(pre["ticker"], pre["current_value"], strict=False))
    cash_after = float(cash_balance)
    total_cost = 0.0
    gross_traded = 0.0

    for row in pre.itertuples(index=False):
        ticker = str(row.ticker)
        target_weight = float(target_lookup.get(ticker, 0.0))
        current_weight = float(row.current_value) / pre_value
        weight_diff = target_weight - current_weight
        is_tradable = _is_tradable(ticker, execution_ts, unavailable, tradability_callback)
        trade_value = (
            0.0
            if abs(weight_diff) < resolved.trade_band or not is_tradable
            else weight_diff * pre_value
        )
        cost = calculate_transaction_cost(trade_value, resolved.cost_config)
        total_cost += cost
        gross_traded += abs(trade_value)
        post_values[ticker] = float(row.current_value) + trade_value
        cash_after -= trade_value + cost
        trade_rows.append(
            {
                "signal_date": signal_ts,
                "execution_date": execution_ts,
                "ticker": ticker,
                "execution_price_type": resolved.execution_price,
                "execution_price": float(row.execution_price),
                "shares_before": float(row.shares),
                "current_value": float(row.current_value),
                "current_weight": current_weight,
                "target_weight": target_weight,
                "weight_diff": weight_diff,
                "trade_value": trade_value,
                "trade_shares": trade_value / float(row.execution_price),
                "side": _trade_side(trade_value),
                "transaction_cost": cost,
                "is_new_entry": float(row.shares) == 0.0 and target_weight > 0.0,
                "is_exit": float(row.shares) > 0.0 and target_weight == 0.0,
                "is_tradable": is_tradable,
                "trade_reason": _trade_reason(row.shares, target_weight, trade_value, is_tradable),
            }
        )

    turnover = gross_traded / pre_value
    post_value = pre_value - total_cost
    post_weights = _post_trade_weights(post_values, cash_after, post_value)
    trades = pd.DataFrame(trade_rows).sort_values("ticker").reset_index(drop=True)
    return RebalanceResult(
        trades=trades,
        turnover=turnover,
        transaction_cost=total_cost,
        post_trade_weights=post_weights,
        cash_balance=cash_after,
        signal_date=signal_ts,
        execution_date=execution_ts,
    )


def get_month_end_signal_dates(price_data: pd.DataFrame) -> list[pd.Timestamp]:
    """Return monthly last trading dates from a price DataFrame."""
    normalized = _normalize_prices(price_data)
    dates = pd.Series(sorted(normalized["date"].unique()))
    month_end = dates.loc[dates.dt.to_period("M").ne(dates.dt.to_period("M").shift(-1))]
    return [pd.Timestamp(value).normalize() for value in month_end.tolist()]


def _normalize_positions(current_positions: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "shares"}
    missing = required - set(current_positions.columns)
    if missing:
        raise ValueError(f"Missing current position columns: {', '.join(sorted(missing))}")
    positions = current_positions.copy()
    positions["ticker"] = positions["ticker"].astype(str).str.zfill(6)
    positions["shares"] = pd.to_numeric(positions["shares"], errors="raise").astype(float)
    return positions.loc[positions["ticker"] != "CASH", ["ticker", "shares"]]


def _normalize_targets(target_weights: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "target_weight"}
    missing = required - set(target_weights.columns)
    if missing:
        raise ValueError(f"Missing target weight columns: {', '.join(sorted(missing))}")
    targets = target_weights.copy()
    targets["ticker"] = targets["ticker"].astype(str)
    targets.loc[targets["ticker"] != "CASH", "ticker"] = targets.loc[
        targets["ticker"] != "CASH", "ticker"
    ].str.zfill(6)
    targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="raise").astype(float)
    if abs(float(targets["target_weight"].sum()) - 1.0) > 1e-9:
        raise ValueError("target weights must sum to 1.0")
    return targets[["ticker", "target_weight"]]


def _normalize_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "close"}
    missing = required - set(price_data.columns)
    if missing:
        raise ValueError(f"Missing price data columns: {', '.join(sorted(missing))}")
    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    prices["open"] = pd.to_numeric(prices["open"], errors="raise").astype(float)
    prices["close"] = pd.to_numeric(prices["close"], errors="raise").astype(float)
    return prices.sort_values(["date", "ticker"])


def _next_execution_date(price_data: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Timestamp:
    prices = _normalize_prices(price_data)
    future_dates = sorted(
        date for date in prices["date"].unique() if pd.Timestamp(date) > signal_date
    )
    if not future_dates:
        raise ValueError(f"No execution date after signal_date {signal_date.date()}")
    return pd.Timestamp(future_dates[0]).normalize()


def _execution_prices(
    price_data: pd.DataFrame,
    execution_date: pd.Timestamp,
    execution_price: ExecutionPrice,
) -> pd.DataFrame:
    prices = _normalize_prices(price_data)
    price_column = "open" if execution_price == "next_open" else "close"
    execution = prices.loc[prices["date"] == execution_date, ["ticker", price_column]].copy()
    execution = execution.rename(columns={price_column: "execution_price"})
    if execution.empty:
        raise ValueError(f"No prices available on execution date {execution_date.date()}")
    return execution


def _pre_trade_state(
    tickers: list[str],
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    cash_balance: float,
) -> pd.DataFrame:
    del cash_balance
    base = pd.DataFrame({"ticker": tickers})
    merged = base.merge(positions, on="ticker", how="left").merge(prices, on="ticker", how="left")
    if merged["execution_price"].isna().any():
        missing = merged.loc[merged["execution_price"].isna(), "ticker"].tolist()
        raise ValueError(f"Missing execution prices for: {', '.join(missing)}")
    merged["shares"] = merged["shares"].fillna(0.0)
    merged["current_value"] = merged["shares"] * merged["execution_price"]
    return merged


def _is_tradable(
    ticker: str,
    execution_date: pd.Timestamp,
    unavailable_tickers: set[str],
    tradability_callback: TradabilityCallback | None,
) -> bool:
    if ticker in unavailable_tickers:
        return False
    if tradability_callback is not None:
        return bool(tradability_callback(ticker, execution_date))
    return True


def _trade_side(trade_value: float) -> str:
    if trade_value > 0.0:
        return "BUY"
    if trade_value < 0.0:
        return "SELL"
    return "HOLD"


def _trade_reason(
    shares: float, target_weight: float, trade_value: float, is_tradable: bool
) -> str:
    if not is_tradable:
        return "unavailable"
    if trade_value == 0.0:
        return "inside_trade_band"
    if shares == 0.0 and target_weight > 0.0:
        return "new_entry"
    if shares > 0.0 and target_weight == 0.0:
        return "exit"
    return "rebalance_buy" if trade_value > 0.0 else "rebalance_sell"


def _post_trade_weights(
    post_values: dict[str, float],
    cash_after: float,
    post_value: float,
) -> pd.DataFrame:
    rows = [
        {"ticker": ticker, "post_trade_value": value, "post_trade_weight": value / post_value}
        for ticker, value in sorted(post_values.items())
    ]
    rows.append(
        {
            "ticker": "CASH",
            "post_trade_value": cash_after,
            "post_trade_weight": cash_after / post_value,
        }
    )
    return pd.DataFrame(rows)


__all__ = [
    "ExecutionPrice",
    "RebalanceConfig",
    "RebalanceResult",
    "calculate_rebalance_weights",
    "execute_monthly_rebalance",
    "get_month_end_signal_dates",
]
