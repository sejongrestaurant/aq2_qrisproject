"""Application logging configuration."""

from __future__ import annotations

import sys
from typing import Literal

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logging for command-line jobs, APIs, and pipelines."""
    normalized_level = level.upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=normalized_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        enqueue=True,
    )


__all__ = ["LogLevel", "setup_logging"]
