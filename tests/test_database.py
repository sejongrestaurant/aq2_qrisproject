"""Tests for the SQLAlchemy database layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database.connection import (
    SessionFactory,
    create_engine_from_url,
    create_session_factory,
    session_scope,
)
from src.database.models import Base, FactorScore, Stock
from src.database.repositories import (
    get_stock,
    insert_daily_price,
    upsert_factor_score,
    upsert_stock,
)


@pytest.fixture()
def session_factory(tmp_path: Path) -> SessionFactory:
    """Create an isolated SQLite session factory for each test."""
    database_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine_from_url(database_url)
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def test_stock_save_and_lookup(session_factory: SessionFactory) -> None:
    """Stocks should be saved and retrieved by ticker."""
    with session_scope(session_factory) as session:
        upsert_stock(session, _stock_values())

    with session_scope(session_factory) as session:
        stock = get_stock(session, "005930")

    assert stock is not None
    assert stock.ticker == "005930"
    assert stock.company_name == "삼성전자"


def test_daily_prices_prevent_duplicates(session_factory: SessionFactory) -> None:
    """Daily prices should reject duplicate date and ticker rows."""
    with session_scope(session_factory) as session:
        upsert_stock(session, _stock_values())
        insert_daily_price(session, _daily_price_values())

    with pytest.raises(IntegrityError), session_scope(session_factory) as session:
        insert_daily_price(session, _daily_price_values())


def test_factor_scores_upsert(session_factory: SessionFactory) -> None:
    """Factor score upsert should update an existing date and ticker row."""
    values = _factor_score_values(composite_score=77.5, universe_rank=3)

    with session_scope(session_factory) as session:
        upsert_stock(session, _stock_values())
        first = upsert_factor_score(session, values)
        first_id = first.id

    updated_values = _factor_score_values(composite_score=88.25, universe_rank=1)
    with session_scope(session_factory) as session:
        updated = upsert_factor_score(session, updated_values)

    with session_scope(session_factory) as session:
        rows = session.scalars(select(FactorScore)).all()

    assert len(rows) == 1
    assert updated.id == first_id
    assert rows[0].composite_score == 88.25
    assert rows[0].universe_rank == 1


def test_session_scope_rolls_back_on_error(session_factory: SessionFactory) -> None:
    """session_scope should roll back uncommitted writes when an exception occurs."""
    with pytest.raises(RuntimeError), session_scope(session_factory) as session:
        upsert_stock(session, _stock_values())
        raise RuntimeError("force rollback")

    with session_scope(session_factory) as session:
        stocks = session.scalars(select(Stock)).all()

    assert stocks == []


def _stock_values() -> dict[str, object]:
    return {
        "ticker": "005930",
        "company_name": "삼성전자",
        "market": "KOSPI",
        "sector": "반도체",
        "industry": "종합 반도체",
        "investment_theme": "AI, memory",
        "universe_role": "Core",
        "listing_date": date(2010, 1, 1),
        "is_active": True,
    }


def _daily_price_values() -> dict[str, object]:
    return {
        "date": date(2024, 1, 2),
        "ticker": "005930",
        "open": 76000.0,
        "high": 77000.0,
        "low": 75000.0,
        "close": 76500.0,
        "adjusted_close": 76500.0,
        "volume": 12000000.0,
        "trading_value": 918000000000.0,
    }


def _factor_score_values(composite_score: float, universe_rank: int) -> dict[str, object]:
    return {
        "calculation_date": date(2024, 1, 31),
        "ticker": "005930",
        "momentum_raw": 0.12,
        "relative_strength_raw": 0.08,
        "quality_raw": 0.15,
        "growth_raw": 0.07,
        "low_volatility_raw": -0.11,
        "liquidity_raw": 0.2,
        "momentum_score": 80.0,
        "relative_strength_score": 75.0,
        "quality_score": 82.0,
        "growth_score": 70.0,
        "low_volatility_score": 60.0,
        "liquidity_score": 90.0,
        "composite_score": composite_score,
        "universe_rank": universe_rank,
    }
