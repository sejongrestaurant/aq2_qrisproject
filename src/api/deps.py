"""FastAPI dependency helpers."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from src.database.connection import create_engine_from_url, create_session_factory

_engine = create_engine_from_url()
_session_factory = create_session_factory(_engine)


def get_db_session() -> Generator[Session]:
    """Yield a database session for API request handling."""
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


__all__ = ["get_db_session"]
