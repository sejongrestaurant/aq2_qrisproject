"""Seed the local SQLite database with dashboard-friendly development data."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy.orm import Session

from src.config.settings import get_settings
from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.init_db import create_tables
from src.database.models import BacktestDaily, FactorScore, MarketRegime, PortfolioWeight, Stock

ModelT = TypeVar("ModelT")

FACTOR_DATE = date(2026, 6, 30)
REBALANCE_DATE = date(2026, 7, 1)
STRATEGY_NAME = "MUST30 score_weight"


def seed(database_url: str | None = None, universe_csv: Path | None = None) -> None:
    """Create tables and insert deterministic local sample data."""
    create_tables(database_url)
    settings = get_settings()
    csv_path = universe_csv or settings.universe_csv_path
    rows = _read_universe(csv_path)

    engine = create_engine_from_url(database_url)
    session_factory = create_session_factory(engine)
    with session_scope(session_factory) as session:
        _seed_stocks(session, rows)
        _seed_factors(session, rows)
        _seed_regimes(session)
        _seed_portfolio(session, rows[:30])
        _seed_backtest(session)


def _read_universe(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _seed_stocks(session: Session, rows: list[dict[str, str]]) -> None:
    for row in rows:
        _upsert(
            session,
            Stock,
            {"ticker": row["ticker"].zfill(6)},
            {
                "company_name": row["company_name"],
                "market": row["market"],
                "sector": row["sector"],
                "industry": row["industry"],
                "investment_theme": row["investment_theme"],
                "universe_role": row["universe_role"],
                "listing_date": _parse_date(row["data_start_date"], date(2010, 1, 1)),
                "is_active": row["is_active"].lower() == "true",
            },
        )


def _seed_factors(session: Session, rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows, start=1):
        rank_score = max(0.0, 1.0 - (index - 1) / max(len(rows), 1))
        cycle = math.sin(index / 5)
        composite = round(0.45 + rank_score * 1.55 + cycle * 0.08, 6)
        _upsert(
            session,
            FactorScore,
            {"calculation_date": FACTOR_DATE, "ticker": row["ticker"].zfill(6)},
            {
                "momentum_raw": round(0.05 + cycle * 0.04, 6),
                "relative_strength_raw": round(0.5 + rank_score * 0.4, 6),
                "quality_raw": round(0.35 + rank_score * 0.5, 6),
                "growth_raw": round(0.25 + abs(cycle) * 0.4, 6),
                "low_volatility_raw": round(0.25 + (1 - rank_score) * 0.25, 6),
                "liquidity_raw": round(0.45 + rank_score * 0.5, 6),
                "momentum_score": round(0.4 + rank_score * 0.5, 6),
                "relative_strength_score": round(0.35 + rank_score * 0.55, 6),
                "quality_score": round(0.3 + rank_score * 0.6, 6),
                "growth_score": round(0.25 + abs(cycle) * 0.5, 6),
                "low_volatility_score": round(0.25 + (1 - rank_score) * 0.45, 6),
                "liquidity_score": round(0.4 + rank_score * 0.5, 6),
                "composite_score": composite,
                "universe_rank": index,
            },
        )


def _seed_regimes(session: Session) -> None:
    start = date(2025, 1, 31)
    for month_index in range(18):
        current = _add_months(start, month_index)
        regime = "Risk-On" if month_index % 6 < 3 else "Neutral" if month_index % 6 < 5 else "Risk-Off"
        score = 3.0 if regime == "Risk-On" else 0.0 if regime == "Neutral" else -3.0
        close = 2550 + month_index * 38 + (40 if regime == "Risk-On" else -70 if regime == "Risk-Off" else 0)
        _upsert(
            session,
            MarketRegime,
            {"date": current},
            {
                "regime": regime,
                "kospi_close": round(close, 2),
                "moving_average": round(2500 + month_index * 34, 2),
                "volatility": round(0.14 + (0.08 if regime == "Risk-Off" else 0.02), 4),
                "market_breadth": 0.62 if regime == "Risk-On" else 0.5 if regime == "Neutral" else 0.36,
                "score": score,
            },
        )


def _seed_portfolio(session: Session, rows: list[dict[str, str]]) -> None:
    total_score = sum(max(0.2, 1.0 - index / 45) for index, _ in enumerate(rows))
    for index, row in enumerate(rows, start=1):
        score = max(0.2, 1.0 - (index - 1) / 45)
        target_weight = Decimal(str(round(score / total_score, 8)))
        _upsert(
            session,
            PortfolioWeight,
            {"rebalance_date": REBALANCE_DATE, "ticker": row["ticker"].zfill(6)},
            {
                "target_weight": target_weight,
                "rank": index,
                "regime": "Risk-On",
                "selection_reason": f"{row['company_name']} selected from the composite factor ranking.",
            },
        )


def _seed_backtest(session: Session) -> None:
    current = date(2025, 1, 2)
    value = 100.0
    benchmark_value = 100.0
    peak = value
    index = 0
    while current <= FACTOR_DATE:
        if current.weekday() < 5:
            daily_return = 0.00055 + math.sin(index / 11) * 0.004 + math.cos(index / 23) * 0.0015
            benchmark_return = 0.00035 + math.sin(index / 13) * 0.003
            turnover = 0.28 if index % 21 == 0 and index > 0 else 0.0
            transaction_cost = turnover * 0.002
            value *= 1 + daily_return - transaction_cost
            benchmark_value *= 1 + benchmark_return
            peak = max(peak, value)
            drawdown = value / peak - 1
            _upsert(
                session,
                BacktestDaily,
                {"date": current, "strategy_name": STRATEGY_NAME},
                {
                    "daily_return": round(daily_return - transaction_cost, 8),
                    "portfolio_value": round(value, 6),
                    "benchmark_return": round(benchmark_return, 8),
                    "benchmark_value": round(benchmark_value, 6),
                    "drawdown": round(drawdown, 8),
                    "turnover": turnover,
                    "transaction_cost": round(transaction_cost, 8),
                    "cash_weight": 0.15,
                },
            )
            index += 1
        current += timedelta(days=1)


def _upsert(
    session: Session,
    model: type[ModelT],
    keys: dict[str, Any],
    values: dict[str, Any],
) -> ModelT:
    row = session.query(model).filter_by(**keys).one_or_none()
    if row is None:
        row = model(**keys, **values)  # type: ignore[call-arg]
        session.add(row)
        return row
    for key, value in values.items():
        setattr(row, key, value)
    return row


def _parse_date(value: str, fallback: date) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return fallback


def _add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    return date(year, month, min(value.day, 28))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local development data for the dashboard.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--universe-csv", type=Path, default=None)
    args = parser.parse_args()
    seed(args.database_url, args.universe_csv)


if __name__ == "__main__":
    main()
