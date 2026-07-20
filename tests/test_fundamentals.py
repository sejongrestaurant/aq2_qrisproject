"""Tests for DART fundamental collection and point-in-time access."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.collectors.dart_collector import DartCompany
from src.database.connection import (
    SessionFactory,
    create_engine_from_url,
    create_session_factory,
    session_scope,
)
from src.database.models import Base
from src.database.repositories import upsert_fundamental, upsert_stock
from src.pipeline.collect_fundamentals import collect_fundamentals
from src.preprocessing.fundamental_cleaner import (
    clean_dart_fundamentals,
    convert_cumulative_quarters_to_single_period,
    infer_report_date,
)
from src.preprocessing.point_in_time import get_latest_available_fundamentals


@dataclass
class MockDartCollector:
    """Network-free DART collector."""

    calls: list[tuple[str, int, str]] = field(default_factory=list)

    def fetch_corp_code_mapping(self) -> dict[str, DartCompany]:
        return {
            "005930": DartCompany("00126380", "005930", "삼성전자"),
            **{
                f"{100000 + index:06d}": DartCompany(
                    f"{index:08d}",
                    f"{100000 + index:06d}",
                    f"Company {index}",
                )
                for index in range(1, 100)
            },
        }

    def fetch_financial_statement(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> pd.DataFrame:
        self.calls.append((corp_code, fiscal_year, report_code))
        if report_code not in {"11013", "11011"}:
            return pd.DataFrame()
        return _raw_dart_accounts()

    def fetch_available_date(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> date | None:
        if report_code == "11013":
            return date(fiscal_year, 5, 15)
        if report_code == "11011":
            return date(fiscal_year + 1, 3, 20)
        return None


def test_clean_dart_fundamentals_maps_accounts_and_units() -> None:
    """DART account names should be mapped to project standard fields."""
    record, missing = clean_dart_fundamentals(
        _raw_dart_accounts(),
        ticker="5930",
        fiscal_year=2024,
        report_code="11013",
        report_date=infer_report_date(2024, "11013"),
        available_date=date(2024, 5, 15),
    )

    assert record["ticker"] == "005930"
    assert record["fiscal_quarter"] == 1
    assert record["revenue"] == Decimal("1000")
    assert record["shares_outstanding"] == Decimal("5969782550")
    assert missing == []


def test_convert_cumulative_quarters_to_single_period() -> None:
    """Cumulative flow fields should become single-quarter values."""
    df = pd.DataFrame(
        [
            {
                "ticker": "005930",
                "fiscal_year": 2024,
                "fiscal_quarter": 1,
                "revenue": Decimal("100"),
            },
            {
                "ticker": "005930",
                "fiscal_year": 2024,
                "fiscal_quarter": 2,
                "revenue": Decimal("250"),
            },
            {
                "ticker": "005930",
                "fiscal_year": 2024,
                "fiscal_quarter": 3,
                "revenue": Decimal("450"),
            },
        ]
    )

    converted = convert_cumulative_quarters_to_single_period(df)

    assert converted["revenue"].tolist() == [Decimal("100"), Decimal("150"), Decimal("200")]


def test_point_in_time_excludes_unavailable_corrections(tmp_path: Path) -> None:
    """A correction should not be visible before its actual available_date."""
    session_factory = _session_factory(tmp_path)
    with session_scope(session_factory) as session:
        upsert_stock(session, _stock_values())
        base_values = _fundamental_values(
            report_date=date(2024, 3, 31),
            available_date=date(2024, 5, 15),
            revenue=Decimal("1000"),
        )
        correction_values = _fundamental_values(
            report_date=date(2024, 3, 31),
            available_date=date(2024, 6, 1),
            revenue=Decimal("1200"),
        )
        upsert_fundamental(session, base_values)
        upsert_fundamental(session, correction_values)

    with session_scope(session_factory) as session:
        before_correction = get_latest_available_fundamentals(
            "005930", date(2024, 5, 31), session=session
        )
        after_correction = get_latest_available_fundamentals(
            "005930", date(2024, 6, 1), session=session
        )

    assert before_correction is not None
    assert after_correction is not None
    assert before_correction["revenue"] == Decimal("1000.0000")
    assert after_correction["revenue"] == Decimal("1200.0000")


def test_collect_fundamentals_with_mock_collector(tmp_path: Path) -> None:
    """Pipeline should collect mock DART data and upsert fundamentals without network calls."""
    universe_path = _write_universe(tmp_path)
    stats = collect_fundamentals(
        universe_path=universe_path,
        start_year=2024,
        end_year=2024,
        database_url=f"sqlite:///{tmp_path / 'fundamentals.db'}",
        failure_output_path=tmp_path / "failures.csv",
        missing_account_output_path=tmp_path / "missing.csv",
        collector=MockDartCollector(),
    )

    assert stats.target_count == 100
    assert stats.success_count == 100
    assert stats.saved_rows == 200
    assert stats.missing_account_count == 0


def _raw_dart_accounts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"account_nm": "매출액", "thstrm_amount": "1,000"},
            {"account_nm": "영업이익", "thstrm_amount": "200"},
            {"account_nm": "당기순이익", "thstrm_amount": "150"},
            {"account_nm": "자산총계", "thstrm_amount": "5,000"},
            {"account_nm": "자본총계", "thstrm_amount": "3,000"},
            {"account_nm": "부채총계", "thstrm_amount": "2,000"},
            {"account_nm": "영업활동현금흐름", "thstrm_amount": "180"},
            {"account_nm": "발행주식수", "thstrm_amount": "5,969,782,550"},
        ]
    )


def _session_factory(tmp_path: Path) -> SessionFactory:
    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'pit.db'}")
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def _stock_values() -> dict[str, object]:
    return {
        "ticker": "005930",
        "company_name": "삼성전자",
        "market": "KOSPI",
        "sector": "Technology",
        "industry": "Semiconductor",
        "investment_theme": "AI",
        "universe_role": "Core",
        "listing_date": date(2010, 1, 1),
        "is_active": True,
    }


def _fundamental_values(
    *,
    report_date: date,
    available_date: date,
    revenue: Decimal,
) -> dict[str, object]:
    return {
        "ticker": "005930",
        "report_date": report_date,
        "available_date": available_date,
        "fiscal_year": 2024,
        "fiscal_quarter": 1,
        "revenue": revenue,
        "operating_income": Decimal("200"),
        "net_income": Decimal("150"),
        "total_assets": Decimal("5000"),
        "total_equity": Decimal("3000"),
        "total_debt": Decimal("2000"),
        "operating_cash_flow": Decimal("180"),
        "shares_outstanding": Decimal("5969782550"),
    }


def _write_universe(tmp_path: Path) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(100):
        ticker = "005930" if index == 0 else f"{100000 + index:06d}"
        rows.append(
            {
                "rank": index + 1,
                "ticker": ticker,
                "company_name": "삼성전자" if index == 0 else f"Company {index}",
                "market": "KOSPI" if index % 2 == 0 else "KOSDAQ",
                "sector": "Technology",
                "industry": "Software",
                "investment_theme": "Theme",
                "universe_role": "Core",
                "selection_reason": "Test row",
                "data_start_date": "2020-01-01",
                "is_active": True,
                "notes": "Test",
            }
        )

    universe_path = tmp_path / "universe.csv"
    pd.DataFrame(rows).to_csv(universe_path, index=False)
    return universe_path
