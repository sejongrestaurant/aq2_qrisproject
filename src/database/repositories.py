"""Repository helpers for database writes and reads."""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.database.models import DailyPrice, FactorScore, Fundamental, Stock


def get_stock(session: Session, ticker: str) -> Stock | None:
    """Return one stock by ticker."""
    return session.get(Stock, ticker)


def upsert_stock(session: Session, values: dict[str, Any]) -> Stock:
    """Insert or update a stock by primary-key ticker."""
    ticker = str(values["ticker"]).zfill(6)
    normalized_values = {**values, "ticker": ticker}
    stock = session.get(Stock, ticker)

    if stock is None:
        stock = Stock(**normalized_values)
        session.add(stock)
        session.flush()
        return stock

    _apply_values(stock, normalized_values)
    session.flush()
    return stock


def insert_daily_price(session: Session, values: dict[str, Any]) -> DailyPrice:
    """Insert a daily price row and rely on the database unique constraint for duplicates."""
    daily_price = DailyPrice(**values)
    session.add(daily_price)
    session.flush()
    return daily_price


def upsert_daily_price(session: Session, values: dict[str, Any]) -> DailyPrice:
    """Insert or update a daily price row by date and ticker."""
    existing = _get_daily_price(session, values["date"], values["ticker"])
    if existing is None:
        return insert_daily_price(session, values)

    _apply_values(existing, values)
    session.flush()
    return existing


def upsert_daily_prices(session: Session, rows: list[dict[str, Any]]) -> int:
    """Insert or update many daily price rows and return the affected row count."""
    for row in rows:
        upsert_daily_price(session, row)
    return len(rows)


def get_latest_daily_price_date(session: Session, ticker: str) -> date | None:
    """Return the latest stored daily price date for one ticker."""
    statement = select(func.max(DailyPrice.date)).where(DailyPrice.ticker == ticker)
    return session.scalar(statement)


def delete_daily_prices(
    session: Session,
    ticker: str,
    start_date: date,
    end_date: date,
) -> int:
    """Delete daily prices for one ticker in an inclusive date range."""
    statement = delete(DailyPrice).where(
        DailyPrice.ticker == ticker,
        DailyPrice.date >= start_date,
        DailyPrice.date <= end_date,
    )
    result = session.execute(statement)
    return int(result.rowcount or 0)


def upsert_factor_score(session: Session, values: dict[str, Any]) -> FactorScore:
    """Insert or update a factor score row by calculation date and ticker."""
    existing = _get_factor_score(session, values["calculation_date"], values["ticker"])
    if existing is None:
        factor_score = FactorScore(**values)
        session.add(factor_score)
        session.flush()
        return factor_score

    _apply_values(existing, values)
    session.flush()
    return existing


def upsert_fundamental(session: Session, values: dict[str, Any]) -> Fundamental:
    """Insert or update point-in-time fundamentals by ticker, report date, and available date."""
    existing = _get_fundamental(
        session,
        ticker=values["ticker"],
        report_date=values["report_date"],
        available_date=values["available_date"],
    )
    if existing is None:
        fundamental = Fundamental(**values)
        session.add(fundamental)
        session.flush()
        return fundamental

    _apply_values(existing, values)
    session.flush()
    return existing


def upsert_fundamentals(session: Session, rows: list[dict[str, Any]]) -> int:
    """Insert or update many point-in-time fundamental rows."""
    for row in rows:
        upsert_fundamental(session, row)
    return len(rows)


def get_latest_available_fundamental(
    session: Session,
    ticker: str,
    as_of_date: date,
) -> Fundamental | None:
    """Return the latest fundamental record actually available by as_of_date."""
    statement = (
        select(Fundamental)
        .where(
            Fundamental.ticker == ticker,
            Fundamental.available_date <= as_of_date,
        )
        .order_by(Fundamental.available_date.desc(), Fundamental.report_date.desc())
        .limit(1)
    )
    return session.scalars(statement).one_or_none()


def _get_daily_price(session: Session, price_date: date, ticker: str) -> DailyPrice | None:
    statement = select(DailyPrice).where(DailyPrice.date == price_date, DailyPrice.ticker == ticker)
    return session.scalars(statement).one_or_none()


def _get_factor_score(
    session: Session, calculation_date: date, ticker: str
) -> FactorScore | None:
    statement = select(FactorScore).where(
        FactorScore.calculation_date == calculation_date,
        FactorScore.ticker == ticker,
    )
    return session.scalars(statement).one_or_none()


def _get_fundamental(
    session: Session,
    ticker: str,
    report_date: date,
    available_date: date,
) -> Fundamental | None:
    statement = select(Fundamental).where(
        Fundamental.ticker == ticker,
        Fundamental.report_date == report_date,
        Fundamental.available_date == available_date,
    )
    return session.scalars(statement).one_or_none()


def _apply_values(model: object, values: dict[str, Any]) -> None:
    for key, value in values.items():
        setattr(model, key, value)


__all__ = [
    "get_stock",
    "get_latest_daily_price_date",
    "delete_daily_prices",
    "insert_daily_price",
    "get_latest_available_fundamental",
    "upsert_daily_price",
    "upsert_daily_prices",
    "upsert_factor_score",
    "upsert_fundamental",
    "upsert_fundamentals",
    "upsert_stock",
]
