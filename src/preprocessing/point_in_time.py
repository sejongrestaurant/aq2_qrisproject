"""Point-in-time accessors for fundamentals."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.models import Fundamental
from src.database.repositories import get_latest_available_fundamental


def get_latest_available_fundamentals(
    ticker: str,
    as_of_date: date,
    session: Session | None = None,
) -> dict[str, object] | None:
    """Return the latest financial record that was disclosed by as_of_date."""
    normalized_ticker = str(ticker).zfill(6)
    if session is not None:
        row = get_latest_available_fundamental(session, normalized_ticker, as_of_date)
        return _fundamental_to_dict(row) if row else None

    engine = create_engine_from_url()
    session_factory = create_session_factory(engine)
    with session_scope(session_factory) as managed_session:
        row = get_latest_available_fundamental(managed_session, normalized_ticker, as_of_date)
        return _fundamental_to_dict(row) if row else None


def _fundamental_to_dict(row: Fundamental) -> dict[str, object]:
    return {
        "ticker": row.ticker,
        "report_date": row.report_date,
        "available_date": row.available_date,
        "fiscal_year": row.fiscal_year,
        "fiscal_quarter": row.fiscal_quarter,
        "revenue": _decimal_or_none(row.revenue),
        "operating_income": _decimal_or_none(row.operating_income),
        "net_income": _decimal_or_none(row.net_income),
        "total_assets": _decimal_or_none(row.total_assets),
        "total_equity": _decimal_or_none(row.total_equity),
        "total_debt": _decimal_or_none(row.total_debt),
        "operating_cash_flow": _decimal_or_none(row.operating_cash_flow),
        "shares_outstanding": _decimal_or_none(row.shares_outstanding),
    }


def _decimal_or_none(value: Decimal | None) -> Decimal | None:
    return value


__all__ = ["get_latest_available_fundamentals"]
