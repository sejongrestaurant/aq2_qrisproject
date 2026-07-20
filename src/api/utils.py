"""API utility functions."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import pandas as pd
from fastapi import HTTPException, Query
from sqlalchemy.orm import Query as SqlAlchemyQuery

from src.api.schemas import PageMeta
from src.config.settings import get_settings
from src.database.models import BacktestDaily


def clean_value(value: object) -> object:
    """Convert NaN, infinities, pandas scalars, and Decimal values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "item") and not isinstance(value, (date, str, bytes)):
        return clean_value(value.item())
    return value


def clean_dict(row: dict[str, object]) -> dict[str, object]:
    """Return a JSON-safe dict without NaN values."""
    return {key: clean_value(value) for key, value in row.items()}


def pagination_params(
    limit: int = Query(default=100, ge=1),
    offset: int = Query(default=0, ge=0),
) -> tuple[int, int]:
    """Validate and cap pagination parameters."""
    settings = get_settings()
    max_size = settings.api_max_page_size
    if limit > max_size:
        raise HTTPException(status_code=413, detail=f"limit exceeds maximum page size {max_size}")
    return limit, offset


def page_meta(limit: int, offset: int, count: int) -> PageMeta:
    """Build pagination metadata."""
    return PageMeta(limit=limit, offset=offset, count=count)


def apply_date_range(
    query: SqlAlchemyQuery,
    model_date: object,
    start_date: date | None,
    end_date: date | None,
) -> SqlAlchemyQuery:
    """Apply inclusive date filters to a SQLAlchemy query."""
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be <= end_date")
    if start_date is not None:
        query = query.filter(model_date >= start_date)
    if end_date is not None:
        query = query.filter(model_date <= end_date)
    return query


def require_strategy_exists(
    session: object, model: type[BacktestDaily], strategy_name: str
) -> None:
    """Raise 404 when a strategy has no stored backtest rows."""
    exists = session.query(model).filter(model.strategy_name == strategy_name).first()
    if exists is None:
        raise HTTPException(status_code=404, detail=f"strategy not found: {strategy_name}")


__all__ = [
    "apply_date_range",
    "clean_dict",
    "clean_value",
    "page_meta",
    "pagination_params",
    "require_strategy_exists",
]
