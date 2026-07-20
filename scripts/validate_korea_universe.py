"""Validate the generated Korea active ETF universe CSV."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "data" / "universe" / "korea_active_etf_universe_100.csv"

REQUIRED_COLUMNS = [
    "rank",
    "ticker",
    "company_name",
    "market",
    "sector",
    "industry",
    "investment_theme",
    "universe_role",
    "selection_reason",
    "data_start_date",
    "is_active",
    "notes",
]

VALID_MARKETS = {"KOSPI", "KOSDAQ"}
VALID_ROLES = {"Core", "Growth", "Defensive", "Cyclical"}
MAX_SECTOR_WEIGHT = 0.25


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows using UTF-8-SIG encoding."""
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                "CSV 컬럼 순서가 요구사항과 다릅니다: "
                f"expected={REQUIRED_COLUMNS}, actual={reader.fieldnames}"
            )
        return list(reader)


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Validate row count, uniqueness, domain values, missing values, and sector caps."""
    errors: list[str] = []

    if len(rows) != 100:
        errors.append(f"전체 종목 수가 정확히 100개가 아닙니다: {len(rows)}")

    ranks: list[int] = []
    for item in rows:
        try:
            ranks.append(int(item["rank"]))
        except ValueError:
            errors.append(f"rank가 정수가 아닙니다: {item['rank']}")
    if ranks != list(range(1, 101)):
        errors.append("rank가 1부터 100까지 연속적이지 않습니다.")

    tickers = [item["ticker"] for item in rows]
    bad_tickers = [ticker for ticker in tickers if len(ticker) != 6 or not ticker.isdigit()]
    if bad_tickers:
        errors.append(f"6자리 문자열이 아닌 ticker가 있습니다: {bad_tickers}")

    duplicate_tickers = [ticker for ticker, count in Counter(tickers).items() if count > 1]
    if duplicate_tickers:
        errors.append(f"ticker 중복이 있습니다: {duplicate_tickers}")

    names = [item["company_name"] for item in rows]
    duplicate_names = [name for name, count in Counter(names).items() if count > 1]
    if duplicate_names:
        errors.append(f"company_name 중복이 있습니다: {duplicate_names}")

    invalid_markets = sorted({item["market"] for item in rows} - VALID_MARKETS)
    if invalid_markets:
        errors.append(f"market 값이 KOSPI 또는 KOSDAQ이 아닙니다: {invalid_markets}")

    invalid_roles = sorted({item["universe_role"] for item in rows} - VALID_ROLES)
    if invalid_roles:
        errors.append(f"universe_role 값이 허용 범위가 아닙니다: {invalid_roles}")

    for row_number, item in enumerate(rows, start=2):
        missing_columns = [column for column in REQUIRED_COLUMNS if item.get(column) in ("", None)]
        if missing_columns:
            errors.append(f"CSV {row_number}행 필수 컬럼 결측치: {missing_columns}")

    if rows:
        sector_counts = Counter(item["sector"] for item in rows)
        over_limit = {
            sector: count
            for sector, count in sector_counts.items()
            if count / len(rows) > MAX_SECTOR_WEIGHT
        }
        if over_limit:
            errors.append(f"단일 sector 비중이 25%를 초과했습니다: {over_limit}")

    if errors:
        raise ValueError("\n".join(errors))


def main() -> None:
    """Run standalone CSV validation."""
    try:
        rows = load_rows(CSV_PATH)
        validate_rows(rows)
    except Exception as exc:  # noqa: BLE001 - command-line script should print clear failures.
        raise SystemExit(f"검증 실패:\n{exc}") from exc

    sector_counts = Counter(item["sector"] for item in rows)
    market_counts = Counter(item["market"] for item in rows)
    print("검증 통과")
    print(f"종목 수: {len(rows)}")
    print(f"시장 분포: {dict(market_counts)}")
    print(f"섹터 분포: {dict(sector_counts)}")


if __name__ == "__main__":
    main()
