"""TrendScore V2 — 원본(TrendScoreIndicator)을 상속한 변형 실험용 지표.

원본 코드는 수정하지 않는다. 변형 요소는 전부 생성자 파라미터로 노출해
같은 백테스터에 V1/V2 를 나란히 꽂아 비교할 수 있게 한다.

변형 손잡이:
  reversal_weight — TSMOM 의 '1개월 리버설' 차감 강도.
      1.0 = 원본과 동일(12M 모멘텀 − 1M 수익).
      0.0 = 리버설 차감 제거(순수 12M 모멘텀).
      가설: 리버설 항이 횡단면 선별에서 '식어가는 종목'을 우대하고
      V자 재진입을 늦춘다 → 낮추면 전환 구간 성과 개선 기대, 대신
      과열 진입·베어랠리 오탐 위험 증가.
  smooth_span — 점수 EMA 스무딩(원본에 이미 있는 미사용 옵션, 예: 30).
      가설: 전환 구간(5슬롯대) 휩쏘 완화.
그 외 가중치·문턱은 원본 생성자 파라미터를 그대로 물려받는다.
"""
from __future__ import annotations

import pandas as pd

from .trend_score import TrendScoreIndicator


class TrendScoreV2(TrendScoreIndicator):
    """리버설 차감 강도를 파라미터화한 TrendScore 변형."""

    def __init__(self, reversal_weight: float = 1.0, name: str | None = None, **kwargs):
        label = name or f"TrendScoreV2(rev={reversal_weight}" + (
            f",smooth={kwargs['smooth_span']})" if kwargs.get("smooth_span") else ")")
        super().__init__(name=label, **kwargs)
        self.reversal_weight = float(reversal_weight)

    def _tsmom(self, close: pd.Series, vol: pd.Series) -> pd.Series:
        """원본 공식에서 1개월 리버설 차감에 가중을 건다.

        원본: norm = (r12 − r1) / vol → 본 변형: (r12 − w·r1) / vol
        """
        r12 = (close.shift(22) - close.shift(252)) / (close.shift(252) + 1e-9)
        r1 = (close - close.shift(22)) / (close.shift(22) + 1e-9)
        norm = ((r12 - self.reversal_weight * r1) / vol).clip(-2, 2)
        return (norm + 2.0) / 4.0 * 100.0
