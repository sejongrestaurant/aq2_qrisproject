"""Tests for the look-ahead-safe backtest engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.backtest.engine import BacktestConfig, run_backtest
from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.models import BacktestDaily, Base


def test_backtest_generates_daily_outputs_and_next_day_rebalance() -> None:
    """The engine should store daily values and execute after the signal date."""
    result = run_backtest(
        _prices(),
        _universe(),
        _factors(),
        _market_data(),
        config=BacktestConfig(start_date="2024-01-30", end_date="2024-02-05", target_size=2),
    )

    assert not result.daily_results.empty
    assert not result.monthly_results.empty
    assert not result.holdings.empty
    assert result.trades["signal_date"].min() == pd.Timestamp("2024-01-31")
    assert result.trades["execution_date"].min() == pd.Timestamp("2024-02-01")
    assert not result.trades["execution_date"].eq(result.trades["signal_date"]).any()
    assert result.daily_results["transaction_cost"].sum() > 0.0


def test_backtest_uses_default_commission_and_market_impact_split() -> None:
    """Backtest costs should use the 0.0015 commission and 0.0005 impact split."""
    result = run_backtest(
        _prices(),
        _universe(),
        _factors(),
        _market_data(),
        config=BacktestConfig(start_date="2024-01-30", end_date="2024-02-05", target_size=2),
    )

    traded = result.trades.loc[result.trades["trade_value"].abs() > 0.0].copy()
    expected = traded["trade_value"].abs() * (0.0015 + 0.0005)

    assert abs(float(traded["transaction_cost"].sum()) - float(expected.sum())) <= 1e-9
    assert result.metadata["commission"] == 0.0015
    assert result.metadata["market_impact"] == 0.0005


def test_backtest_uses_point_in_time_universe_and_factor_availability() -> None:
    """Future listings and future-available factor rows must not enter past selection."""
    result = run_backtest(
        _prices(),
        _universe(),
        _factors(),
        _market_data(),
        config=BacktestConfig(start_date="2024-01-30", end_date="2024-02-05", target_size=2),
    )

    selected = result.factor_scores.loc[
        result.factor_scores["signal_date"] == pd.Timestamp("2024-01-31")
    ]
    assert set(selected["ticker"]) == {"000001", "000002"}
    score = selected.loc[selected["ticker"] == "000001", "composite_score"].iloc[0]
    assert score == 1.0


def test_suspended_stock_is_valued_with_last_price_and_not_traded() -> None:
    """Suspended stocks should use the previous price and be passed as untradable."""
    prices = _prices()
    prices.loc[
        (prices["date"] == pd.Timestamp("2024-02-01")) & (prices["ticker"] == "000001"),
        "is_suspended",
    ] = True

    result = run_backtest(
        prices,
        _universe(),
        _factors(),
        _market_data(),
        config=BacktestConfig(start_date="2024-01-30", end_date="2024-02-05", target_size=2),
    )
    trade = result.trades.loc[result.trades["ticker"] == "000001"].iloc[0]

    assert bool(trade["is_tradable"]) is False
    assert trade["trade_value"] == 0.0
    holding = result.holdings.loc[
        (result.holdings["date"] == pd.Timestamp("2024-02-01"))
        & (result.holdings["ticker"] == "000001")
    ]
    assert holding.empty


def test_backtest_is_deterministic_for_same_parameters() -> None:
    """Identical inputs and metadata timestamp should produce identical tables."""
    config = BacktestConfig(
        start_date="2024-01-30",
        end_date="2024-02-05",
        target_size=2,
        created_at="2026-01-01T00:00:00+00:00",
    )
    first = run_backtest(_prices(), _universe(), _factors(), _market_data(), config=config)
    second = run_backtest(_prices(), _universe(), _factors(), _market_data(), config=config)

    pd.testing.assert_frame_equal(first.daily_results, second.daily_results)
    pd.testing.assert_frame_equal(first.trades, second.trades)
    assert first.metadata == second.metadata


def test_backtest_saves_parquet_and_database(tmp_path: Path) -> None:
    """Backtest outputs should be persisted to parquet files and DB daily rows."""
    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'backtest.db'}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    output_dir = tmp_path / "parquet"

    with session_scope(session_factory) as session:
        result = run_backtest(
            _prices(),
            _universe(),
            _factors(),
            _market_data(),
            config=BacktestConfig(start_date="2024-01-30", end_date="2024-02-05", target_size=2),
            output_dir=output_dir,
            db_session=session,
        )

    assert (output_dir / "daily_results.parquet").exists()
    assert (output_dir / "metadata.json").exists()
    assert str(result.metadata["price_policy"]).startswith("Valuation uses adjusted_close")
    with session_scope(session_factory) as session:
        rows = session.scalars(select(BacktestDaily)).all()
    assert len(rows) == len(result.daily_results)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-30", "2024-02-05")
    rows = []
    for date in dates:
        for ticker, base in {"000001": 100.0, "000002": 80.0, "000003": 50.0}.items():
            step = (date - dates[0]).days
            close = base + step
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close + 0.5,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1000.0,
                    "trading_value": 1_000_000.0,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["000001", "000002", "000003"],
            "market": ["KOSPI", "KOSPI", "KOSPI"],
            "sector": ["A", "B", "C"],
            "universe_role": ["Core", "Core", "Growth"],
            "listing_date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2024-02-15"),
            ],
            "data_start_date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2024-02-15"),
            ],
            "is_active": [True, True, True],
        }
    )


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _factor("2024-01-31", "2024-01-31", "000001", 1.0),
            _factor("2024-01-31", "2024-01-31", "000002", 2.0),
            _factor("2024-01-31", "2024-01-31", "000003", 999.0),
            _factor("2024-01-31", "2024-02-02", "000001", 500.0),
        ]
    )


def _factor(
    calculation_date: str, available_date: str, ticker: str, score: float
) -> dict[str, object]:
    return {
        "calculation_date": pd.Timestamp(calculation_date),
        "available_date": pd.Timestamp(available_date),
        "ticker": ticker,
        "composite_score": score,
        "momentum_score": score,
    }


def _market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-30", "2024-02-05")
    return pd.DataFrame(
        {
            "date": dates,
            "kospi_close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "kospi_ma200": [99.0, 100.0, 101.0, 102.0, 103.0],
            "kospi_momentum_60d": [0.01] * 5,
            "kospi_volatility_20d": [0.1] * 5,
            "market_breadth": [0.60] * 5,
        }
    )
