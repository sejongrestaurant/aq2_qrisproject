"""일봉 TrendScore 지표 (0~100).

참고 트리(`utils/indicator_util.calculate_trend_score_series` / `etf_universe.trend_score`)의
일봉 TrendScore 를 OOP 로 재구성한다. 네 요소를 합성한다:

  1. EWMAC 앙상블 (0.55) — (8/32, 16/64, 32/128) 3쌍의 이동평균 교차를 변동성으로 정규화한 추세.
  2. TSMOM       (0.25) — 12개월 모멘텀에서 1개월 리버설을 뺀 시계열 모멘텀.
  3. RSI         (0.20) — 과열/침체.
  4. ADX penalty        — 추세 강도(|ADX|)가 약하면 최대 15점 차감(횡보장 오탐 억제).
  (옵션) Macro overlay   — 매크로 확률 딕셔너리로 0.5~1.5 배 스케일.

252봉 미만 구간은 NaN(워밍업). 임계 기반 레짐 판정(BULL/…/STRONG_BEAR)도 제공한다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .adx import ADXIndicator
from .base import Indicator
from .rsi import RSIIndicator

# 참고 트리 amoeba `get_regime_from_indicators` 동일 임계
_REGIME_THRESHOLDS = [(62, "BULL"), (50, "WEAK_BULL"), (36, "NEUTRAL"), (20, "BEAR")]


class TrendScoreIndicator(Indicator):
    """일봉 TrendScore(0~100) 계산기.

    Args (생성자):
        min_len: 유효 점수를 내기 위한 최소 봉 수(기본 252). 미만 구간은 NaN.
        rsi_period / adx_period: 구성 지표 기간.
        ewmac_weight / tsmom_weight / rsi_weight: 세 요소 합성 가중(합≈1.0).
        adx_penalty_max: ADX 약세 시 최대 차감 점수(기본 15).
        adx_full_strength: 페널티가 0 이 되는 |ADX| 기준(기본 25).
    """

    def __init__(
        self,
        min_len: int = 252,
        rsi_period: int = 14,
        adx_period: int = 14,
        ewmac_weight: float = 0.55,
        tsmom_weight: float = 0.25,
        rsi_weight: float = 0.20,
        adx_penalty_max: float = 15.0,
        adx_full_strength: float = 25.0,
        smooth_span: int | None = None,
        name: str | None = None,
    ):
        super().__init__(name or "TrendScore")
        self.min_len = min_len
        self.ewmac_weight = ewmac_weight
        self.tsmom_weight = tsmom_weight
        self.rsi_weight = rsi_weight
        self.adx_penalty_max = adx_penalty_max
        self.adx_full_strength = adx_full_strength
        self.smooth_span = smooth_span   # EMA 스무딩 span(거래일). None=원시. 급변 whipsaw 완화(6주≈30).
        self._rsi = RSIIndicator(rsi_period)
        self._adx = ADXIndicator(adx_period)

    # ── public ──────────────────────────────────────────────────────
    def compute(self, data: pd.DataFrame, macro_probs: Optional[dict] = None) -> pd.Series:
        """TrendScore 전체 시계열(0~100, 워밍업 NaN)을 계산한다.

        Args:
            data: 표준 스키마 시세(close 필수, high/low 없으면 close 로 대체).
            macro_probs: 선택적 매크로 확률 딕셔너리(equity/liquidity_signal/fx_signal/volatility).
                        없으면 오버레이 미적용(scale=1.0).
        """
        close = self._col(data, "close", "Close", "adj_close")
        if close is None:
            raise ValueError("TrendScoreIndicator: 'close' 컬럼 필요")
        close = self._series(close)
        high = self._series(self._col(data, "high", "High") if self._col(data, "high", "High") is not None else close)
        low = self._series(self._col(data, "low", "Low") if self._col(data, "low", "Low") is not None else close)

        # 연율화 변동성(하한으로 0 나눗셈 방지) — 모든 정규화의 분모
        log_ret = np.log(close).diff()
        vol = (log_ret.rolling(60).std() * np.sqrt(252)).clip(lower=1e-9)

        ewmac_score = self._ewmac_ensemble(close, vol)
        tsmom_score = self._tsmom(close, vol)
        rsi_score = self._rsi.from_close(close)

        trend_raw = (self.ewmac_weight * ewmac_score
                     + self.tsmom_weight * tsmom_score
                     + self.rsi_weight * rsi_score)

        trend_macro = trend_raw * self._macro_scale(macro_probs)

        adx = self._adx.from_ohlc(high, low, close)
        penalty = self.adx_penalty_max * (1.0 - (adx.abs() / self.adx_full_strength).clip(0, 1))
        final = (trend_macro - penalty).clip(0, 100)

        # 급변 whipsaw 완화용 EMA 스무딩(선택). 랭킹·문턱 판정 전에 노이즈 제거.
        if self.smooth_span:
            final = final.ewm(span=self.smooth_span).mean()

        # 워밍업 마스킹: min_len 미만 구간 NaN
        mask = pd.Series(True, index=close.index)
        mask.iloc[: self.min_len - 1] = False
        return final.where(mask).rename(self.name)

    @classmethod
    def regime(cls, score: Optional[float]) -> str:
        """단일 점수를 레짐 문자열로 변환(BULL/WEAK_BULL/NEUTRAL/BEAR/STRONG_BEAR).

        None/NaN 은 NEUTRAL. 참고 트리 amoeba 임계와 동일.
        """
        if score is None or score != score:  # None/NaN
            return "NEUTRAL"
        s = float(score)
        for threshold, label in _REGIME_THRESHOLDS:
            if s >= threshold:
                return label
        return "STRONG_BEAR"

    # ── 내부 구성요소 ────────────────────────────────────────────────
    def _ewmac_ensemble(self, close: pd.Series, vol: pd.Series) -> pd.Series:
        """(8/32, 16/64, 32/128) EWMAC 3쌍 앙상블 → 0~100 스케일."""
        def _pair(fast: int, slow: int) -> pd.Series:
            raw = (close.ewm(span=fast).mean() - close.ewm(span=slow).mean()) / (close + 1e-9)
            return (raw / vol).clip(-2, 2)

        raw = 0.33 * _pair(8, 32) + 0.33 * _pair(16, 64) + 0.34 * _pair(32, 128)
        return (raw.clip(-2, 2) + 2.0) / 4.0 * 100.0

    def _tsmom(self, close: pd.Series, vol: pd.Series) -> pd.Series:
        """12개월 모멘텀 − 1개월 리버설, 변동성 정규화 → 0~100 스케일."""
        r12 = (close.shift(22) - close.shift(252)) / (close.shift(252) + 1e-9)
        r1 = (close - close.shift(22)) / (close.shift(22) + 1e-9)
        norm = ((r12 - r1) / vol).clip(-2, 2)
        return (norm + 2.0) / 4.0 * 100.0

    @staticmethod
    def _macro_scale(macro_probs: Optional[dict]) -> float:
        """매크로 확률 딕셔너리를 0.5~1.5 배 스케일로 변환(없으면 1.0)."""
        if not macro_probs:
            return 1.0
        raw = (0.5 * macro_probs.get("equity", 0.5)
               + 0.2 * macro_probs.get("liquidity_signal", 0.5)
               + 0.2 * macro_probs.get("fx_signal", 0.5)
               + 0.1 * (1.0 - macro_probs.get("volatility", 0.5)))
        return float(np.clip(raw * 2.0, 0.5, 1.5))
