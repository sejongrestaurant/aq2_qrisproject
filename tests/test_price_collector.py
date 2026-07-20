"""Tests for daily price collection and cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.collectors.base_collector import BasePriceCollector
from src.database.connection import create_engine_from_url, create_session_factory, session_scope
from src.database.models import DailyPrice
from src.pipeline.collect_prices import collect_prices
from src.preprocessing.price_cleaner import clean_price_data


@dataclass
class MockPriceCollector(BasePriceCollector):
    """Network-free collector used by pipeline tests."""

    calls: list[tuple[str, date, date]] = field(default_factory=list)

    def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        self.calls.append((ticker, start_date, end_date))
        dates = list(pd.date_range(start_date, end_date))
        dates.append(pd.Timestamp(end_date))
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": [1000.0] * len(dates),
                "High": [1100.0] * len(dates),
                "Low": [900.0] * len(dates),
                "Close": [1050.0] * len(dates),
                "Volume": [0.0] * len(dates),
            }
        )


def test_clean_price_data_standardizes_and_marks_suspended_rows() -> None:
    """Cleaner should normalize columns, remove invalid rows, and retain zero-volume rows."""
    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "Open": [1000.0, 1000.0, 1000.0, -1.0],
            "High": [1100.0, 1100.0, 1100.0, 1100.0],
            "Low": [900.0, 900.0, 900.0, 900.0],
            "Close": [1050.0, 1050.0, 0.0, 1050.0],
            "Volume": [10.0, 0.0, 10.0, 10.0],
        }
    )

    cleaned_df, duplicate_count = clean_price_data(raw_df, "5930")

    assert duplicate_count == 1
    assert cleaned_df["ticker"].tolist() == ["005930"]
    assert cleaned_df["date"].tolist() == [date(2024, 1, 2)]
    assert cleaned_df["is_suspended"].tolist() == [True]


def test_collect_prices_uses_later_of_user_start_and_listing_date(tmp_path: Path) -> None:
    """Collection should not request data before each stock data_start_date."""
    universe_path = _write_universe(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'prices.db'}"
    collector = MockPriceCollector()

    stats = collect_prices(
        universe_path=universe_path,
        start_date=date(2014, 1, 1),
        end_date=date(2020, 1, 3),
        database_url=database_url,
        failure_output_path=tmp_path / "failures.csv",
        collector=collector,
    )

    first_call = collector.calls[0]

    assert stats.target_count == 100
    assert stats.success_count == 100
    assert stats.failure_count == 0
    assert first_call == ("005930", date(2020, 1, 2), date(2020, 1, 3))


def test_collect_prices_incremental_load_skips_existing_dates(tmp_path: Path) -> None:
    """A second run through the same end date should not call the collector again."""
    universe_path = _write_universe(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'prices.db'}"
    collector = MockPriceCollector()

    collect_prices(
        universe_path=universe_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 2),
        database_url=database_url,
        failure_output_path=tmp_path / "failures.csv",
        collector=collector,
    )
    first_run_calls = len(collector.calls)

    second_stats = collect_prices(
        universe_path=universe_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 2),
        database_url=database_url,
        failure_output_path=tmp_path / "failures.csv",
        collector=collector,
    )

    assert first_run_calls == 100
    assert len(collector.calls) == first_run_calls
    assert second_stats.inserted_rows == 0
    assert second_stats.success_count == 100


def test_collect_prices_stores_suspended_status(tmp_path: Path) -> None:
    """Zero-volume rows should be stored and marked as suspended."""
    universe_path = _write_universe(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'prices.db'}"

    collect_prices(
        universe_path=universe_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 2),
        database_url=database_url,
        failure_output_path=tmp_path / "failures.csv",
        collector=MockPriceCollector(),
    )

    engine = create_engine_from_url(database_url)
    session_factory = create_session_factory(engine)
    with session_scope(session_factory) as session:
        first_price = session.scalars(select(DailyPrice).order_by(DailyPrice.ticker)).first()

    assert first_price is not None
    assert first_price.is_suspended is True


def test_collect_prices_records_failures_without_stopping(tmp_path: Path) -> None:
    """A failed ticker should be recorded while later tickers continue."""

    class FailingOnceCollector(MockPriceCollector):
        def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
            if ticker == "005930":
                raise RuntimeError("mock failure")
            return super().fetch_daily_prices(ticker, start_date, end_date)

    universe_path = _write_universe(tmp_path)
    failure_output_path = tmp_path / "failures.csv"

    stats = collect_prices(
        universe_path=universe_path,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 2),
        database_url=f"sqlite:///{tmp_path / 'prices.db'}",
        failure_output_path=failure_output_path,
        collector=FailingOnceCollector(),
    )

    failures_df = pd.read_csv(failure_output_path)

    assert stats.failure_count == 1
    assert stats.success_count == 99
    assert failures_df["ticker"].astype(str).str.zfill(6).tolist() == ["005930"]


def _write_universe(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(100):
        ticker = "005930" if index == 0 else f"{100000 + index:06d}"
        rows.append(
            {
                "rank": index + 1,
                "ticker": ticker,
                "company_name": f"Company {index}",
                "market": "KOSPI" if index % 2 == 0 else "KOSDAQ",
                "sector": "Technology",
                "industry": "Software",
                "investment_theme": "Theme",
                "universe_role": "Core",
                "selection_reason": "Test row",
                "data_start_date": "2020-01-02" if index == 0 else "2020-01-01",
                "is_active": True,
                "notes": "Test",
            }
        )

    universe_path = tmp_path / "universe.csv"
    pd.DataFrame(rows).to_csv(universe_path, index=False)
    return universe_path
