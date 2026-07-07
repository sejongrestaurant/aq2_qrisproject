"""SuperTrend 지표 (ATR 밴드 추세추종).

가격의 중앙값(HL2)에 ATR×배수 밴드를 씌워, 추세 방향을 따라 이동하는 트레일링 스탑 라인을 만든다.
종가가 라인 위면 상승추세(롱), 아래면 하락추세(플랫). 오실레이터(0~100)가 아니라 **가격 수준 지표**라,
리포트에서는 가격 차트에 밴드 라인으로 겹쳐 그린다.

TrendScore(합성 점수형)와 대비되는 **밴드 돌파형** 추세 신호로, 스윙 타이밍 비교의 대조군이 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator


class SuperTrendIndicator(Indicator):
    """SuperTrend 라인 + 방향 계산기.

    Args (생성자):
        atr_period: ATR 평활 기간(기본 10).
        multiplier: ATR 밴드 배수(기본 3.0). 클수록 둔감(휩쏘↓, 반응 느림).
    """

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0, name: str | None = None):
        super().__init__(name or f"SuperTrend({atr_period},{multiplier:g})")
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """SuperTrend 라인(가격 수준) Series 를 반환한다."""
        return self.compute_with_direction(data)[0]

    def compute_with_direction(self, data: pd.DataFrame):
        """SuperTrend 라인과 방향을 함께 계산한다.

        Returns:
            (supertrend: pd.Series 가격 수준 라인, uptrend: pd.Series[bool] 상승추세 여부).
            워밍업(atr_period 미만) 구간은 라인 NaN, uptrend False.
        """
        high = self._series(self._col(data, "high", "High"))
        low = self._series(self._col(data, "low", "Low"))
        close = self._series(self._col(data, "close", "Close", "adj_close"))
        if close is None or high is None or low is None:
            raise ValueError("SuperTrendIndicator: high/low/close 컬럼 필요")

        atr = self._atr(high, low, close, self.atr_period)
        hl2 = (high + low) / 2.0
        upper_basic = hl2 + self.multiplier * atr
        lower_basic = hl2 - self.multiplier * atr

        n = len(close)
        c = close.to_numpy()
        ub, lb = upper_basic.to_numpy(), lower_basic.to_numpy()
        final_ub = np.full(n, np.nan)
        final_lb = np.full(n, np.nan)
        st = np.full(n, np.nan)
        up = np.zeros(n, dtype=bool)

        warm = self.atr_period  # ATR 유효 시작 인덱스
        for i in range(n):
            if i < warm or np.isnan(ub[i]):
                continue
            if i == warm or np.isnan(final_ub[i - 1]):
                # 밴드·추세 초기화(첫 유효 봉)
                final_ub[i], final_lb[i] = ub[i], lb[i]
                st[i] = final_ub[i]
                up[i] = c[i] > st[i]
                continue

            # 최종 밴드 캐리포워드 규칙(추세 방향으로만 좁혀짐)
            final_ub[i] = (ub[i] if (ub[i] < final_ub[i - 1] or c[i - 1] > final_ub[i - 1])
                           else final_ub[i - 1])
            final_lb[i] = (lb[i] if (lb[i] > final_lb[i - 1] or c[i - 1] < final_lb[i - 1])
                           else final_lb[i - 1])

            # 라인/방향 전환 판정
            if st[i - 1] == final_ub[i - 1]:            # 직전 하락추세(라인=상단)
                st[i] = final_ub[i] if c[i] <= final_ub[i] else final_lb[i]
            else:                                        # 직전 상승추세(라인=하단)
                st[i] = final_lb[i] if c[i] >= final_lb[i] else final_ub[i]
            up[i] = st[i] == final_lb[i]

        idx = close.index
        return (pd.Series(st, index=idx, name=self.name),
                pd.Series(up, index=idx, name="uptrend"))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Wilder ATR(참 범위의 지수이동평균)."""
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()
