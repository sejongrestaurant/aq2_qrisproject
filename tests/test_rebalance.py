"""Tests for monthly rebalance execution."""

from __future__ import annotations

import pandas as pd

from src.backtest.costs import TransactionCostConfig, calculate_transaction_cost
from src.portfolio.rebalance import (
    RebalanceConfig,
    execute_monthly_rebalance,
    get_month_end_signal_dates,
)


def test_rebalance_executes_on_next_trading_day_not_signal_day() -> None:
    """Signal-day close should not be used as the execution date."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
    )

    assert result.signal_date == pd.Timestamp("2024-01-31")
    assert result.execution_date == pd.Timestamp("2024-02-01")
    assert result.trades["execution_date"].eq(pd.Timestamp("2024-02-01")).all()
    assert not result.trades["execution_date"].eq(pd.Timestamp("2024-01-31")).any()


def test_pre_rebalance_drifted_weights_are_calculated() -> None:
    """Current weights should be based on execution-date prices before trading."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
    )
    row = result.trades.loc[result.trades["ticker"] == "000001"].iloc[0]

    pre_value = 100 * 110 + 100 * 90 + 50_000
    assert abs(float(row["current_weight"]) - (11_000 / pre_value)) <= 1e-12


def test_trade_band_skips_small_weight_differences() -> None:
    """Positions within the 1 percentage point band should not trade."""
    prices = _prices()
    targets = pd.DataFrame(
        {
            "ticker": ["000001", "000002", "CASH"],
            "target_weight": [0.158, 0.132, 0.710],
        }
    )

    result = execute_monthly_rebalance(
        _positions(),
        targets,
        prices,
        "2024-01-31",
        cash_balance=50_000.0,
    )

    assert result.trades["trade_value"].eq(0.0).all()
    assert result.turnover == 0.0
    assert result.transaction_cost == 0.0


def test_turnover_buy_sell_amounts_and_transaction_costs() -> None:
    """Buy and sell values should both contribute to turnover and costs."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
    )
    traded = result.trades.loc[result.trades["trade_value"] != 0.0]
    expected_cost = traded["trade_value"].abs().sum() * 0.002
    pre_value = 100 * 110 + 100 * 90 + 50_000

    assert (traded["side"] == "BUY").any()
    assert (traded["side"] == "SELL").any()
    assert abs(result.turnover - float(traded["trade_value"].abs().sum() / pre_value)) <= 1e-12
    assert abs(result.transaction_cost - float(expected_cost)) <= 1e-12
    assert abs(calculate_transaction_cost(10_000.0) - 20.0) <= 1e-12


def test_cost_is_deducted_from_portfolio_value_and_cash() -> None:
    """Post-trade value should equal pre-trade value minus transaction costs."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
    )
    pre_value = 100 * 110 + 100 * 90 + 50_000
    post_value = float(result.post_trade_weights["post_trade_value"].sum())

    assert abs(post_value - (pre_value - result.transaction_cost)) <= 1e-9
    assert abs(float(result.post_trade_weights["post_trade_weight"].sum()) - 1.0) <= 1e-12
    assert (
        result.cash_balance
        == result.post_trade_weights.loc[
            result.post_trade_weights["ticker"] == "CASH",
            "post_trade_value",
        ].iloc[0]
    )


def test_unavailable_ticker_interface_blocks_trade() -> None:
    """Unavailable or untradable tickers should be retained but not traded."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
        unavailable_tickers={"000001"},
    )
    row = result.trades.loc[result.trades["ticker"] == "000001"].iloc[0]

    assert bool(row["is_tradable"]) is False
    assert row["trade_value"] == 0.0
    assert row["trade_reason"] == "unavailable"


def test_new_entry_and_exit_are_identified() -> None:
    """Trade details should distinguish newly added and removed holdings."""
    positions = pd.DataFrame({"ticker": ["000001", "000003"], "shares": [100.0, 100.0]})
    targets = pd.DataFrame(
        {
            "ticker": ["000001", "000002", "CASH"],
            "target_weight": [0.20, 0.20, 0.60],
        }
    )

    result = execute_monthly_rebalance(
        positions,
        targets,
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
    )

    assert (
        result.trades.loc[result.trades["ticker"] == "000002", "trade_reason"].iloc[0]
        == "new_entry"
    )
    assert result.trades.loc[result.trades["ticker"] == "000003", "trade_reason"].iloc[0] == "exit"


def test_next_close_execution_price_is_optional() -> None:
    """Execution price can be switched from next_open to next_close."""
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
        config=RebalanceConfig(execution_price="next_close"),
    )
    row = result.trades.loc[result.trades["ticker"] == "000001"].iloc[0]

    assert row["execution_price_type"] == "next_close"
    assert row["execution_price"] == 111.0


def test_month_end_signal_dates_are_monthly_last_trading_days() -> None:
    """Monthly signal helper should emit one last trading date per month."""
    dates = get_month_end_signal_dates(_prices())

    assert dates == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-02")]


def test_custom_cost_config_is_applied() -> None:
    """Commission and market impact assumptions should be configurable."""
    config = RebalanceConfig(
        cost_config=TransactionCostConfig(commission_rate=0.001, market_impact_rate=0.001)
    )
    result = execute_monthly_rebalance(
        _positions(),
        _targets(),
        _prices(),
        "2024-01-31",
        cash_balance=50_000.0,
        config=config,
    )
    traded_abs = result.trades["trade_value"].abs().sum()

    assert abs(result.transaction_cost - float(traded_abs * 0.002)) <= 1e-12


def _positions() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["000001", "000002"], "shares": [100.0, 100.0]})


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["000001", "000002", "CASH"],
            "target_weight": [0.25, 0.05, 0.70],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _price("2024-01-31", "000001", 100.0, 100.0),
            _price("2024-01-31", "000002", 100.0, 100.0),
            _price("2024-01-31", "000003", 100.0, 100.0),
            _price("2024-02-01", "000001", 110.0, 111.0),
            _price("2024-02-01", "000002", 90.0, 89.0),
            _price("2024-02-01", "000003", 50.0, 49.0),
            _price("2024-02-02", "000001", 112.0, 113.0),
            _price("2024-02-02", "000002", 91.0, 92.0),
            _price("2024-02-02", "000003", 51.0, 52.0),
        ]
    )


def _price(date: str, ticker: str, open_price: float, close_price: float) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "ticker": ticker,
        "open": open_price,
        "close": close_price,
    }
