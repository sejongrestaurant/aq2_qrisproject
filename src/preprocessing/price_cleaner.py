"""Price data normalization and cleaning."""

from __future__ import annotations

from datetime import date
from typing import Final

import pandas as pd

STANDARD_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "trading_value",
    "is_suspended",
)

COLUMN_RENAMES: Final[dict[str, str]] = {
    "날짜": "date",
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "수정종가": "adjusted_close",
    "거래량": "volume",
    "거래대금": "trading_value",
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Adj_Close": "adjusted_close",
    "Volume": "volume",
    "Change": "change",
}


def clean_price_data(raw_df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, int]:
    """Normalize one ticker's raw OHLCV DataFrame and return duplicate count removed."""
    if raw_df.empty:
        return pd.DataFrame(columns=STANDARD_PRICE_COLUMNS), 0

    normalized_ticker = _normalize_ticker(ticker)
    df = raw_df.copy()
    df = _ensure_date_column(df)
    df = df.rename(columns=COLUMN_RENAMES)
    df["ticker"] = normalized_ticker

    missing_columns = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required price columns: {', '.join(sorted(missing_columns))}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
    else:
        df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce").fillna(
            df["close"]
        )

    if "trading_value" not in df.columns:
        df["trading_value"] = df["close"] * df["volume"]
    else:
        df["trading_value"] = pd.to_numeric(df["trading_value"], errors="coerce")
        df["trading_value"] = df["trading_value"].fillna(df["close"] * df["volume"])

    before_deduplicate = len(df)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    duplicate_count = before_deduplicate - len(df)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "trading_value",
    ]
    df = df.dropna(subset=["date", *numeric_columns])
    df = df.loc[(df[numeric_columns] >= 0).all(axis=1)].copy()
    df = df.loc[df["close"] > 0].copy()
    df["is_suspended"] = df["volume"] == 0

    return df.loc[:, STANDARD_PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(
        drop=True
    ), duplicate_count


def to_daily_price_records(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert cleaned price rows to repository records."""
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "date": row.date,
                "ticker": row.ticker,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "adjusted_close": float(row.adjusted_close),
                "volume": float(row.volume),
                "trading_value": float(row.trading_value),
                "is_suspended": bool(row.is_suspended),
            }
        )
    return records


def filter_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter cleaned prices to an inclusive date range."""
    return df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns or "Date" in df.columns or "날짜" in df.columns:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        return df.reset_index(names="date")

    if df.index.name is not None:
        return df.reset_index().rename(columns={df.index.name: "date"})

    raise ValueError("Price data must include a date column or DatetimeIndex")


def _normalize_ticker(ticker: str) -> str:
    normalized_ticker = str(ticker).strip()
    if normalized_ticker.endswith(".0"):
        normalized_ticker = normalized_ticker[:-2]
    if not normalized_ticker.isdigit():
        raise ValueError(f"ticker must contain only digits: {ticker}")
    return normalized_ticker.zfill(6)


__all__ = [
    "STANDARD_PRICE_COLUMNS",
    "clean_price_data",
    "filter_date_range",
    "to_daily_price_records",
]
