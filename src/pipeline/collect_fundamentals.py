"""CLI pipeline for collecting DART fundamentals."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd
from loguru import logger

from src.collectors.dart_collector import ALL_REPORT_CODES, DartCollector, DartCompany
from src.config.settings import get_settings
from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.init_db import create_tables
from src.database.repositories import upsert_fundamentals, upsert_stock
from src.preprocessing.fundamental_cleaner import (
    clean_dart_fundamentals,
    convert_cumulative_quarters_to_single_period,
    infer_report_date,
)
from src.universe.loader import get_active_universe, load_universe


class FundamentalCollector(Protocol):
    """Collector interface used by the fundamentals pipeline."""

    def fetch_corp_code_mapping(self) -> dict[str, DartCompany]:
        """Return DART corp codes keyed by ticker."""

    def fetch_financial_statement(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> pd.DataFrame:
        """Return raw DART financial statement rows."""

    def fetch_available_date(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> date | None:
        """Return the filing date available to investors."""


@dataclass
class FundamentalCollectionStats:
    """Fundamental collection result counters."""

    target_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    saved_rows: int = 0
    missing_account_count: int = 0
    elapsed_seconds: float = 0.0


def collect_fundamentals(
    *,
    universe_path: Path,
    start_year: int,
    end_year: int,
    database_url: str | None = None,
    failure_output_path: Path = Path("data/raw/fundamental_collection_failures.csv"),
    missing_account_output_path: Path = Path("data/raw/fundamental_missing_accounts.csv"),
    collector: FundamentalCollector | None = None,
) -> FundamentalCollectionStats:
    """Collect annual and quarterly DART fundamentals and persist point-in-time rows."""
    if start_year > end_year:
        raise ValueError(f"start_year must be <= end_year: {start_year} > {end_year}")

    started_at = time.perf_counter()
    settings = get_settings()
    resolved_database_url = database_url or settings.database_url
    dart_collector = collector or DartCollector(api_key=settings.dart_api_key or None)

    create_tables(resolved_database_url)
    engine = create_engine_from_url(resolved_database_url)
    session_factory = create_session_factory(engine)

    universe_df = get_active_universe(load_universe(universe_path))
    corp_mapping = dart_collector.fetch_corp_code_mapping()
    stats = FundamentalCollectionStats(target_count=len(universe_df))
    failures: list[dict[str, str]] = []
    missing_accounts: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    logger.info("Fundamental collection target count: {}", stats.target_count)

    with session_scope(session_factory) as session:
        for row in universe_df.itertuples(index=False):
            ticker = str(row.ticker).zfill(6)
            upsert_stock(session, _stock_values_from_universe_row(row))
            company = corp_mapping.get(ticker)
            if company is None:
                stats.failure_count += 1
                failures.append({"ticker": ticker, "error": "DART corp_code not found"})
                continue

            ticker_success = False
            for fiscal_year in range(start_year, end_year + 1):
                for report_code in ALL_REPORT_CODES:
                    try:
                        raw_df = dart_collector.fetch_financial_statement(
                            company.corp_code,
                            fiscal_year,
                            report_code,
                        )
                        if raw_df.empty:
                            continue

                        available_date = dart_collector.fetch_available_date(
                            company.corp_code,
                            fiscal_year,
                            report_code,
                        )
                        if available_date is None:
                            raise ValueError("DART available_date could not be resolved")

                        record, missing = clean_dart_fundamentals(
                            raw_df,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                            report_code=report_code,
                            report_date=infer_report_date(fiscal_year, report_code),
                            available_date=available_date,
                        )
                        records.append(record)
                        ticker_success = True
                        for account in missing:
                            missing_accounts.append(
                                {
                                    "ticker": ticker,
                                    "corp_code": company.corp_code,
                                    "fiscal_year": fiscal_year,
                                    "report_code": report_code,
                                    "missing_account": account,
                                }
                            )
                    except Exception as error:
                        failures.append(
                            {
                                "ticker": ticker,
                                "corp_code": company.corp_code,
                                "fiscal_year": str(fiscal_year),
                                "report_code": report_code,
                                "error": str(error),
                            }
                        )
                        logger.warning(
                            "{} {} {} failed: {}", ticker, fiscal_year, report_code, error
                        )

            if ticker_success:
                stats.success_count += 1
            else:
                stats.failure_count += 1

        if records:
            normalized_df = convert_cumulative_quarters_to_single_period(pd.DataFrame(records))
            stats.saved_rows = upsert_fundamentals(
                session,
                normalized_df.where(pd.notna(normalized_df), None).to_dict("records"),
            )

    stats.missing_account_count = len(missing_accounts)
    _save_report(failures, failure_output_path)
    _save_report(missing_accounts, missing_account_output_path)
    stats.elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "Fundamental collection finished: target={}, success={}, failure={}, saved_rows={}, missing_accounts={}, elapsed_seconds={:.2f}",
        stats.target_count,
        stats.success_count,
        stats.failure_count,
        stats.saved_rows,
        stats.missing_account_count,
        stats.elapsed_seconds,
    )
    return stats


def main() -> None:
    """Run the DART fundamental collection CLI."""
    parser = argparse.ArgumentParser(description="Collect DART fundamentals.")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument(
        "--universe-path", default="data/universe/korea_active_etf_universe_100.csv"
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--failure-output-path", default="data/raw/fundamental_collection_failures.csv"
    )
    parser.add_argument(
        "--missing-account-output-path",
        default="data/raw/fundamental_missing_accounts.csv",
    )
    args = parser.parse_args()

    stats = collect_fundamentals(
        universe_path=Path(args.universe_path),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        database_url=args.database_url,
        failure_output_path=Path(args.failure_output_path),
        missing_account_output_path=Path(args.missing_account_output_path),
    )

    print(f"target_count: {stats.target_count}")
    print(f"success_count: {stats.success_count}")
    print(f"failure_count: {stats.failure_count}")
    print(f"saved_rows: {stats.saved_rows}")
    print(f"missing_account_count: {stats.missing_account_count}")
    print(f"elapsed_seconds: {stats.elapsed_seconds:.2f}")


def _stock_values_from_universe_row(row: object) -> dict[str, object]:
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


def _save_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
