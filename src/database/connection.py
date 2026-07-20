"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings

SessionFactory = sessionmaker[Session]


def create_engine_from_url(database_url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL-compatible configuration."""
    resolved_url = database_url or get_settings().database_url
    url = make_url(resolved_url)

    if url.drivername.startswith("sqlite") and url.database not in {None, "", ":memory:"}:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.drivername.startswith("sqlite") else {}
    return create_engine(resolved_url, echo=echo, future=True, connect_args=connect_args)


def create_session_factory(engine: Engine) -> SessionFactory:
    """Create a configured SQLAlchemy session factory."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope(session_factory: SessionFactory) -> Generator[Session]:
    """Provide a transactional scope around a series of database operations."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["SessionFactory", "create_engine_from_url", "create_session_factory", "session_scope"]
