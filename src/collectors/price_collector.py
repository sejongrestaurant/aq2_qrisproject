"""Daily price collectors for Korean stocks."""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd

from src.collectors.base_collector import BasePriceCollector, RetryConfig


class PriceCollectionError(RuntimeError):
    """Raised when all price data providers fail."""


class PykrxPriceCollector(BasePriceCollector):
    """Fetch daily OHLCV data from pykrx."""

    def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily prices from pykrx for one ticker."""
        normalized_ticker = _normalize_ticker(ticker)

        def operation() -> pd.DataFrame:
            stock_module = importlib.import_module("pykrx.stock")
            return stock_module.get_market_ohlcv_by_date(
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
                normalized_ticker,
            )

        return self._with_retry(operation, label=f"pykrx:{normalized_ticker}")


class FinanceDataReaderPriceCollector(BasePriceCollector):
    """Fetch daily OHLCV data from FinanceDataReader."""

    def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily prices from FinanceDataReader for one ticker."""
        normalized_ticker = _normalize_ticker(ticker)

        def operation() -> pd.DataFrame:
            fdr_module = importlib.import_module("FinanceDataReader")
            return fdr_module.DataReader(
                normalized_ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

        return self._with_retry(operation, label=f"fdr:{normalized_ticker}")


class PriceCollector(BasePriceCollector):
    """Fetch daily prices using pykrx first and FinanceDataReader as fallback."""

    def __init__(
        self,
        primary: BasePriceCollector | None = None,
        fallback: BasePriceCollector | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(retry_config)
        self.primary = primary or PykrxPriceCollector(self.retry_config)
        self.fallback = fallback or FinanceDataReaderPriceCollector(self.retry_config)

    def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily prices from the primary source and then fallback if needed."""
        normalized_ticker = _normalize_ticker(ticker)
        primary_error: Exception | None = None

        try:
            primary_df = self.primary.fetch_daily_prices(normalized_ticker, start_date, end_date)
            if not primary_df.empty:
                return primary_df
        except Exception as error:
            primary_error = error

        try:
            fallback_df = self.fallback.fetch_daily_prices(normalized_ticker, start_date, end_date)
            if not fallback_df.empty:
                return fallback_df
        except Exception as fallback_error:
            raise PriceCollectionError(
                f"Failed to collect {normalized_ticker} from pykrx and FinanceDataReader"
            ) from fallback_error

        raise PriceCollectionError(
            f"No data returned for {normalized_ticker} from pykrx or FinanceDataReader"
        ) from primary_error


def _normalize_ticker(ticker: str) -> str:
    normalized_ticker = str(ticker).strip()
    if normalized_ticker.endswith(".0"):
        normalized_ticker = normalized_ticker[:-2]
    if not normalized_ticker.isdigit():
        raise ValueError(f"ticker must contain only digits: {ticker}")
    return normalized_ticker.zfill(6)


__all__ = [
    "FinanceDataReaderPriceCollector",
    "PriceCollectionError",
    "PriceCollector",
    "PykrxPriceCollector",
]
