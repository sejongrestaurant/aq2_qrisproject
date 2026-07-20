"""Base collector utilities with retry and rate limiting."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TypeVar

import pandas as pd
from loguru import logger

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RetryConfig:
    """Retry and request delay settings for external data providers."""

    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    request_delay_seconds: float = 0.2


class BasePriceCollector(ABC):
    """Common interface for daily price collectors."""

    def __init__(self, retry_config: RetryConfig | None = None) -> None:
        self.retry_config = retry_config or RetryConfig()

    @abstractmethod
    def fetch_daily_prices(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily prices for one ticker."""

    def _with_retry(self, operation: Callable[[], ResultT], *, label: str) -> ResultT:
        """Run an operation with retries and delay between attempts."""
        last_error: Exception | None = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                result = operation()
                time.sleep(self.retry_config.request_delay_seconds)
                return result
            except Exception as error:
                last_error = error
                logger.warning(
                    "{} failed on attempt {}/{}: {}",
                    label,
                    attempt,
                    self.retry_config.max_attempts,
                    error,
                )
                if attempt < self.retry_config.max_attempts:
                    time.sleep(self.retry_config.retry_delay_seconds)

        raise RuntimeError(
            f"{label} failed after {self.retry_config.max_attempts} attempts"
        ) from last_error


__all__ = ["BasePriceCollector", "RetryConfig"]
