"""3조 전략 포팅 — regime-trendrider_mjs_v4 (풀노출 추세라이딩 + 선제 청산).

원본 Pine v6(`strategy("regime-trendrider_mjs_v4")`)을 프레임워크 롱-플랫 모델로 1:1 이식.
  · 국면: EMA20 vs EMA60 + ADX(14) > 10  → 상승/하락/보합
  · 진입: 상승국면 전환(첫 봉) 또는 상승국면 중 종가가 EMA20 상향돌파
  · 청산(OR): ① 샹들리에 스탑(22고점 − 2.5·ATR) 이탈 ② 하락국면 ③ B1 선제청산
      (EMA20이 3봉 전보다 낮고 종가 < EMA20 — 데드크로스 확정 전 위험 축소)

지표는 Pine 정의에 맞춰 EMA(재귀), Wilder ADX/ATR(rma=ewm α=1/n)을 자체 계산한다.
체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지). 비용은 엔진 공통값 사용(원본 0.5%는 미적용).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signals, Strategy


class RegimeTrendRiderStrategy(Strategy):
    """EMA 국면 + ADX 추세 + 샹들리에/선제 청산 롱-플랫 전략(3조 v4 이식)."""

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 60,
        adx_len: int = 14,
        adx_trend: float = 10.0,
        ch_len: int = 22,
        ch_mult: float = 2.5,
        atr_len: int = 14,
        slope_lb: int = 3,
        name: str | None = None,
    ):
        super().__init__(name or "TrendRider v4 (3조)")
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_len = adx_len
        self.adx_trend = adx_trend
        self.ch_len = ch_len
        self.ch_mult = ch_mult
        self.atr_len = atr_len
        self.slope_lb = slope_lb

    # ── 지표(Pine 정의 대응) ────────────────────────────────────────
    @staticmethod
    def _rma(s: pd.Series, n: int) -> pd.Series:
        """Wilder RMA = Pine ta.rma (ewm α=1/n, adjust=False)."""
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    def _adx(self, high, low, close) -> pd.Series:
        up = high.diff()
        dn = -low.diff()
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = self._rma(tr, self.adx_len)
        plus_di = 100.0 * self._rma(plus_dm, self.adx_len) / atr
        minus_di = 100.0 * self._rma(minus_dm, self.adx_len) / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return self._rma(dx.fillna(0.0), self.adx_len)

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]
        close = data["close"] if "close" in data.columns else data["Close"]

        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        adx = self._adx(high, low, close)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = self._rma(tr, self.atr_len)
        chand = high.rolling(self.ch_len).max() - self.ch_mult * atr

        is_up = (ema_fast > ema_slow) & (adx > self.adx_trend)
        is_down = (ema_fast < ema_slow) & (adx > self.adx_trend)
        crossover = (close > ema_fast) & (close.shift(1) <= ema_fast.shift(1))

        entry = is_up & ((~is_up.shift(1).fillna(False)) | crossover)   # 상승전환 또는 EMA20 돌파
        exit_ = ((close < chand) | is_down                              # 샹들리에 이탈 · 하락국면
                 | ((ema_fast < ema_fast.shift(self.slope_lb)) & (close < ema_fast)))  # B1 선제청산

        en = entry.to_numpy(); ex = exit_.to_numpy()
        n = len(close); state = np.zeros(n, dtype=bool)
        on = False
        for i in range(n):
            if on:
                if ex[i]:
                    on = False
            elif en[i]:
                on = True
            state[i] = on

        target = pd.Series(state, index=close.index, name="target_long")
        return Signals(target_long=target,
                       indicators={f"ADX{self.adx_len}": adx},
                       overlays={f"EMA{self.ema_fast}": ema_fast,
                                 f"EMA{self.ema_slow}": ema_slow,
                                 "Chandelier": chand})
