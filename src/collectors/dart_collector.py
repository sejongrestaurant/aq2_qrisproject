"""DART OpenAPI collector for financial statement data."""

from __future__ import annotations

import os
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import TypeVar
from xml.etree import ElementTree

import pandas as pd
import requests

from src.collectors.base_collector import RetryConfig

ResultT = TypeVar("ResultT")
DART_BASE_URL = "https://opendart.fss.or.kr/api"
ANNUAL_REPORT_CODE = "11011"
QUARTER_REPORT_CODES = ("11013", "11012", "11014")
ALL_REPORT_CODES = (ANNUAL_REPORT_CODE, *QUARTER_REPORT_CODES)


class DartCollectionError(RuntimeError):
    """Raised when a DART API request cannot produce usable data."""


@dataclass(frozen=True)
class DartCompany:
    """DART company code mapping record."""

    corp_code: str
    ticker: str
    corp_name: str


class DartCollector:
    """Collect corp-code mappings and financial statement rows from DART."""

    def __init__(
        self,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DART_API_KEY", "")
        if not self.api_key:
            raise ValueError("DART_API_KEY is required")
        self.retry_config = retry_config or RetryConfig()
        self.timeout_seconds = timeout_seconds

    def fetch_corp_code_mapping(self) -> dict[str, DartCompany]:
        """Fetch listed company corp_code records keyed by six-digit ticker."""
        response = self._request_bytes(
            f"{DART_BASE_URL}/corpCode.xml",
            params={"crtfc_key": self.api_key},
            label="dart:corpCode",
        )
        with zipfile.ZipFile(BytesIO(response)) as archive:
            xml_data = archive.read("CORPCODE.xml")

        root = ElementTree.fromstring(xml_data)
        mapping: dict[str, DartCompany] = {}
        for item in root.findall("list"):
            ticker = (item.findtext("stock_code") or "").strip()
            if not ticker:
                continue
            normalized_ticker = ticker.zfill(6)
            mapping[normalized_ticker] = DartCompany(
                corp_code=(item.findtext("corp_code") or "").strip(),
                ticker=normalized_ticker,
                corp_name=(item.findtext("corp_name") or "").strip(),
            )
        return mapping

    def fetch_financial_statement(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> pd.DataFrame:
        """Fetch financial statements, preferring consolidated statements over separate ones."""
        for fs_div in ("CFS", "OFS"):
            rows = self._request_json(
                f"{DART_BASE_URL}/fnlttSinglAcntAll.json",
                params={
                    "crtfc_key": self.api_key,
                    "corp_code": corp_code,
                    "bsns_year": str(fiscal_year),
                    "reprt_code": report_code,
                    "fs_div": fs_div,
                },
                label=f"dart:financials:{corp_code}:{fiscal_year}:{report_code}:{fs_div}",
            )
            if rows.get("status") == "000" and rows.get("list"):
                df = pd.DataFrame(rows["list"])
                df["fs_div"] = fs_div
                return df

        return pd.DataFrame()

    def fetch_available_date(
        self,
        corp_code: str,
        fiscal_year: int,
        report_code: str,
    ) -> date | None:
        """Fetch the DART receipt date for a report, including corrected filings when listed."""
        response = self._request_json(
            f"{DART_BASE_URL}/list.json",
            params={
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": f"{fiscal_year}0101",
                "end_de": f"{fiscal_year + 1}1231",
                "page_no": "1",
                "page_count": "100",
            },
            label=f"dart:list:{corp_code}:{fiscal_year}:{report_code}",
        )
        if response.get("status") != "000":
            return None

        candidates = [
            item
            for item in response.get("list", [])
            if _report_name_matches_code(str(item.get("report_nm", "")), report_code)
        ]
        if not candidates:
            return None

        latest = max(str(item["rcept_dt"]) for item in candidates if item.get("rcept_dt"))
        return date(int(latest[:4]), int(latest[4:6]), int(latest[6:8]))

    def _request_json(self, url: str, params: dict[str, str], label: str) -> dict[str, object]:
        def operation() -> dict[str, object]:
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            return dict(response.json())

        return _with_retry(operation, self.retry_config, label)

    def _request_bytes(self, url: str, params: dict[str, str], label: str) -> bytes:
        def operation() -> bytes:
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            return bytes(response.content)

        return _with_retry(operation, self.retry_config, label)


def save_corp_code_mapping(mapping: dict[str, DartCompany], path: Path) -> None:
    """Save a corp-code mapping for audit/debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ticker": company.ticker,
                "corp_code": company.corp_code,
                "corp_name": company.corp_name,
            }
            for company in mapping.values()
        ]
    ).sort_values("ticker").to_csv(path, index=False, encoding="utf-8-sig")


def _with_retry[ResultT](
    operation: Callable[[], ResultT],
    retry_config: RetryConfig,
    label: str,
) -> ResultT:
    last_error: Exception | None = None
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            result = operation()
            time.sleep(retry_config.request_delay_seconds)
            return result
        except Exception as error:
            last_error = error
            if attempt < retry_config.max_attempts:
                time.sleep(retry_config.retry_delay_seconds)
    raise DartCollectionError(
        f"{label} failed after {retry_config.max_attempts} attempts"
    ) from last_error


def _report_name_matches_code(report_name: str, report_code: str) -> bool:
    if report_code == "11011":
        return "사업보고서" in report_name
    if report_code == "11013":
        return "1분기" in report_name
    if report_code == "11012":
        return "반기" in report_name
    if report_code == "11014":
        return "3분기" in report_name
    return False


__all__ = [
    "ALL_REPORT_CODES",
    "ANNUAL_REPORT_CODE",
    "DartCollector",
    "DartCompany",
    "DartCollectionError",
    "QUARTER_REPORT_CODES",
    "save_corp_code_mapping",
]
