"""CLI pipeline for collecting daily stock prices."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from src.collectors.base_collector import BasePriceCollector
from src.collectors.price_collector import PriceCollector
from src.config.settings import get_settings
from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.init_db import create_tables
from src.database.repositories import (
    delete_daily_prices,
    get_latest_daily_price_date,
    upsert_daily_prices,
    upsert_stock,
)
from src.preprocessing.price_cleaner import (
    clean_price_data,
    filter_date_range,
    to_daily_price_records,
)
from src.universe.loader import get_active_universe, load_universe

DEFAULT_START_DATE = date(2014, 1, 1)


@dataclass
class CollectionStats:
    """Price collection result counters."""

    target_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    inserted_rows: int = 0
    duplicate_removed_rows: int = 0
    elapsed_seconds: float = 0.0


class UniverseRow(Protocol):
    """Typed subset of a loaded universe row used by the collection pipeline."""

    ticker: str
    company_name: str
    market: str
    sector: str
    industry: str
    investment_theme: str
    universe_role: str
    data_start_date: object
    is_active: bool


def collect_prices(
    *,
    universe_path: Path,
    start_date: date,
    end_date: date,
    database_url: str | None = None,
    full_refresh: bool = False,
    save_raw_parquet: bool = False,
    raw_output_dir: Path = Path("data/raw/prices"),
    failure_output_path: Path = Path("data/raw/price_collection_failures.csv"),
    collector: BasePriceCollector | None = None,
) -> CollectionStats:
    """Collect daily prices for the active universe and save them to the database."""
    if start_date > end_date:
        raise ValueError(f"start_date must be on or before end_date: {start_date} > {end_date}")

    started_at = time.perf_counter()
    settings = get_settings()
    resolved_database_url = database_url or settings.database_url
    price_collector = collector or PriceCollector()

    create_tables(resolved_database_url)
    engine = create_engine_from_url(resolved_database_url)
    session_factory = create_session_factory(engine)

    universe_df = get_active_universe(load_universe(universe_path))
    stats = CollectionStats(target_count=len(universe_df))
    failures: list[dict[str, str]] = []

    logger.info("Price collection target count: {}", stats.target_count)

    for row in universe_df.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        stock_start_date = pd.Timestamp(row.data_start_date).date()
        requested_start_date = max(start_date, stock_start_date)

        try:
            with session_scope(session_factory) as session:
                upsert_stock(session, _stock_values_from_universe_row(row))
                effective_start_date = _resolve_incremental_start_date(
                    session=session,
                    ticker=ticker,
                    requested_start_date=requested_start_date,
                    end_date=end_date,
                    full_refresh=full_refresh,
                )

                if effective_start_date is None:
                    logger.info("{} skipped; already collected through {}", ticker, end_date)
                    stats.success_count += 1
                    continue

                if full_refresh:
                    delete_daily_prices(session, ticker, effective_start_date, end_date)

            with session_scope(session_factory) as session:
                raw_df = price_collector.fetch_daily_prices(ticker, effective_start_date, end_date)
                if save_raw_parquet:
                    _save_raw_parquet(
                        raw_df, raw_output_dir, ticker, effective_start_date, end_date
                    )

                cleaned_df, duplicate_count = clean_price_data(raw_df, ticker)
                cleaned_df = filter_date_range(cleaned_df, effective_start_date, end_date)
                records = to_daily_price_records(cleaned_df)
                stats.inserted_rows += upsert_daily_prices(session, records)
                stats.duplicate_removed_rows += duplicate_count
                stats.success_count += 1

                logger.info(
                    "{} collected from {} to {}; saved rows={}",
                    ticker,
                    effective_start_date,
                    end_date,
                    len(records),
                )
        except Exception as error:
            stats.failure_count += 1
            failures.append({"ticker": ticker, "error": str(error)})
            logger.exception("{} collection failed: {}", ticker, error)

    _save_failures(failures, failure_output_path)
    stats.elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "Price collection finished: target={}, success={}, failure={}, new_rows={}, duplicates_removed={}, elapsed_seconds={:.2f}",
        stats.target_count,
        stats.success_count,
        stats.failure_count,
        stats.inserted_rows,
        stats.duplicate_removed_rows,
        stats.elapsed_seconds,
    )
    return stats


def main() -> None:
    """Run the price collection CLI."""
    parser = argparse.ArgumentParser(description="Collect Korean stock daily prices.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--universe-path", default="data/universe/korea_active_etf_universe_100.csv"
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--save-raw-parquet", action="store_true")
    parser.add_argument("--raw-output-dir", default="data/raw/prices")
    parser.add_argument("--failure-output-path", default="data/raw/price_collection_failures.csv")
    args = parser.parse_args()

    stats = collect_prices(
        universe_path=Path(args.universe_path),
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        database_url=args.database_url,
        full_refresh=bool(args.full_refresh),
        save_raw_parquet=bool(args.save_raw_parquet),
        raw_output_dir=Path(args.raw_output_dir),
        failure_output_path=Path(args.failure_output_path),
    )

    print(f"target_count: {stats.target_count}")
    print(f"success_count: {stats.success_count}")
    print(f"failure_count: {stats.failure_count}")
    print(f"new_rows: {stats.inserted_rows}")
    print(f"duplicates_removed: {stats.duplicate_removed_rows}")
    print(f"elapsed_seconds: {stats.elapsed_seconds:.2f}")


def _resolve_incremental_start_date(
    *,
    session: Session,
    ticker: str,
    requested_start_date: date,
    end_date: date,
    full_refresh: bool,
) -> date | None:
    if full_refresh:
        return requested_start_date

    latest_date = get_latest_daily_price_date(session, ticker)
    if latest_date is None:
        return requested_start_date

    incremental_start_date = max(requested_start_date, latest_date + timedelta(days=1))
    if incremental_start_date > end_date:
        return None
    return incremental_start_date


def _stock_values_from_universe_row(row: UniverseRow) -> dict[str, object]:
    return {
        "ticker": str(row.ticker).zfill(6),
        "company_name": row.company_name,
        "market": row.market,
        "sector": row.sector,
        "industry": row.industry,
        "investment_theme": row.investment_theme,
        "universe_role": row.universe_role,
        "listing_date": pd.Timestamp(row.data_start_date).date(),
        "is_active": bool(row.is_active),
    }


def _save_raw_parquet(
    raw_df: pd.DataFrame,
    raw_output_dir: Path,
    ticker: str,
    start_date: date,
    end_date: date,
) -> None:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_output_dir / f"{ticker}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.parquet"
    raw_df.to_parquet(output_path, index=True)


def _save_failures(failures: list[dict[str, str]], failure_output_path: Path) -> None:
    failure_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["ticker", "error"]).to_csv(
        failure_output_path, index=False, encoding="utf-8-sig"
    )


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    main()
