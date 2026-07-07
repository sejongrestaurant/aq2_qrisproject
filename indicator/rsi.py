"""RSI(Relative Strength Index) 지표.

Wilder 방식(지수이동평균, com=period-1)으로 계산한다. 참고 트리 `indicator_util` 과 동일 공식이라
TrendScore 구성요소로 쓸 때 본 시스템 값과 정합한다.
"""
from __future__ import annotations

import pandas as pd

from .base import Indicator


class RSIIndicator(Indicator):
    """종가 기반 RSI(0~100).

    Args (생성자):
        period: 평활 기간(기본 14).
    """

    def __init__(self, period: int = 14, name: str | None = None):
        super().__init__(name or f"RSI{period}")
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = self._col(data, "close", "Close", "adj_close")
        if close is None:
            raise ValueError("RSIIndicator: 'close' 컬럼 필요")
        return self.from_close(self._series(close), self.period)

    @staticmethod
    def from_close(close: pd.Series, period: int = 14) -> pd.Series:
        """종가 Series 로부터 RSI 를 직접 계산(다른 지표에서 재사용)."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
