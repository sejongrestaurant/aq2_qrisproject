"""Create database tables for the project."""

from __future__ import annotations

import argparse

from loguru import logger
from sqlalchemy import Engine, inspect, text

from src.config.settings import get_settings
from src.database.connection import create_engine_from_url
from src.database.models import Base


def create_tables(database_url: str | None = None) -> None:
    """Create all database tables."""
    resolved_url = database_url or get_settings().database_url
    engine = create_engine_from_url(resolved_url)
    Base.metadata.create_all(engine)
    _ensure_compatible_schema(engine)
    logger.info("Database tables created for {}", resolved_url)


def _ensure_compatible_schema(engine: Engine) -> None:
    """Apply small compatibility fixes for local databases created before migrations exist."""
    inspector = inspect(engine)
    if "daily_prices" not in inspector.get_table_names():
        return

    daily_price_columns = {column["name"] for column in inspector.get_columns("daily_prices")}
    if "is_suspended" in daily_price_columns:
        return

    default_value = "false" if engine.dialect.name == "postgresql" else "0"
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE daily_prices "
                f"ADD COLUMN is_suspended BOOLEAN NOT NULL DEFAULT {default_value}"
            )
        )


def main() -> None:
    """Run the table creation CLI."""
    parser = argparse.ArgumentParser(description="Create MUST30 database tables.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL. Defaults to the environment/config value.",
    )
    args = parser.parse_args()
    create_tables(args.database_url)


if __name__ == "__main__":
    main()
