"""SuperTrend 기반 스윙 전략 (롱-플랫).

SuperTrend 라인(ATR 밴드 트레일링 스탑)의 방향을 그대로 매매 신호로 쓴다:
  · 상승추세(종가 > SuperTrend 라인) → 롱 보유,
  · 하락추세(종가 < 라인) → 플랫.

밴드 자체가 트레일링 스탑이라 별도 히스테리시스가 필요 없다(방향 전환이 곧 진입/청산). TrendScore
스윙(합성 점수 + 임계)과 대비되는 **밴드 돌파형** 대조군으로, 동일 백테스트 엔진에서 헤드투헤드 비교된다.

신호는 봉 i 종가 기준 확정, 체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지).
"""
from __future__ import annotations

import pandas as pd

from indicator import SuperTrendIndicator

from .base import Signals, Strategy


class SuperTrendSwingStrategy(Strategy):
    """SuperTrend 방향 추종 롱-플랫 전략.

    Args (생성자):
        atr_period: ATR 기간(기본 10).
        multiplier: ATR 밴드 배수(기본 3.0).
        indicator: 주입할 SuperTrend 지표(없으면 위 파라미터로 생성).
    """

    def __init__(
        self,
        atr_period: int = 10,
        multiplier: float = 3.0,
        indicator: SuperTrendIndicator | None = None,
        name: str | None = None,
    ):
        super().__init__(name or f"SuperTrend {atr_period}×{multiplier:g}")
        self.indicator = indicator or SuperTrendIndicator(atr_period, multiplier)

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        st_line, uptrend = self.indicator.compute_with_direction(data)
        target = uptrend.rename("target_long")
        # SuperTrend 라인은 가격 수준 지표 → 리포트 가격 차트에 오버레이
        return Signals(target_long=target, overlays={self.indicator.name: st_line})
