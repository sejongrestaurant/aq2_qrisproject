"""1조 전략 포팅 — 국면·시점 (슈퍼트렌드 + 200EMA + Williams %R + ATR 고정손절).

원본 Pine v6(`strategy("국면·시점 전략 · 슈퍼트렌드+200EMA+%R+ATR손절")`)을
프레임워크 롱-플랫 모델로 1:1 이식.
  · 국면(Layer1): 종가>200EMA 이면서 SuperTrend(10,3) 상승 → 상승국면(regimeUp)
  · 시점(Layer2): 상승국면 중 Williams %R(14)이 과매도선(-80)을 상향 돌파(회복) → 진입
  · 청산(OR): ① SuperTrend 상승→하락 반전(stFlipDown)
              ② (옵션) %R 과매수선(-20) 하향 돌파 — useWrExit 기본 False
              ③ ATR 고정손절: 진입가 − 진입시점 ATR × 2.5 (트레일 아님, 진입가 기준 고정)

지표는 Pine 정의에 맞춰 EMA(재귀)·Wilder ATR·Williams %R 을 계산하고,
SuperTrend 방향은 프레임워크 SuperTrendIndicator(동일 TradingView 규칙)를 재사용한다.
체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지). 비용은 엔진 공통값 사용(원본 0.12%는 미적용).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicator import SuperTrendIndicator

from .base import Signals, Strategy


class Team1RegimeStrategy(Strategy):
    """200EMA·SuperTrend 국면 + Williams %R 회복 진입 + ATR 고정손절 롱-플랫 전략(1조 이식)."""

    def __init__(
        self,
        ema_len: int = 200,
        st_atr: int = 10,
        st_factor: float = 3.0,
        wr_len: int = 14,
        wr_os: float = -80.0,
        wr_ob: float = -20.0,
        use_wr_exit: bool = False,
        atr_len: int = 14,
        atr_stop_mult: float = 2.5,
        name: str | None = None,
    ):
        super().__init__(name or "1조 국면·시점 (ST+200EMA+%R+ATR)")
        self.ema_len = ema_len
        self.st = SuperTrendIndicator(st_atr, st_factor)
        self.wr_len = wr_len
        self.wr_os = wr_os
        self.wr_ob = wr_ob
        self.use_wr_exit = use_wr_exit
        self.atr_len = atr_len
        self.atr_stop_mult = atr_stop_mult

    # ── 지표(Pine 정의 대응) ────────────────────────────────────────
    @staticmethod
    def _atr(high, low, close, n) -> pd.Series:
        """Wilder ATR (ta.atr = rma of true range)."""
        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / n, adjust=False).mean()

    def _williams_r(self, high, low, close) -> pd.Series:
        """Williams %R = -100 × (최고가N − 종가) / (최고가N − 최저가N).  범위 -100~0."""
        hh = high.rolling(self.wr_len).max()
        ll = low.rolling(self.wr_len).min()
        rng = (hh - ll).replace(0, np.nan)
        return -100.0 * (hh - close) / rng

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]
        close = data["close"] if "close" in data.columns else data["Close"]

        ema200 = close.ewm(span=self.ema_len, adjust=False).mean()
        st_line, uptrend = self.st.compute_with_direction(data)   # stBull = uptrend
        wr = self._williams_r(high, low, close)
        atr = self._atr(high, low, close, self.atr_len)

        # Layer1 국면
        above_ema = close > ema200
        regime_up = above_ema & uptrend

        # Layer2 시점(신호)
        wr_recover = (wr > self.wr_os) & (wr.shift(1) <= self.wr_os)      # crossover(wr, -80)
        buy_cond = regime_up & wr_recover                                # (+ 플랫 조건은 상태기계가 처리)

        st_flip_down = uptrend.shift(1, fill_value=False) & (~uptrend)    # 상승→하락 반전
        wr_overbought = (self.use_wr_exit
                         & (wr < self.wr_ob) & (wr.shift(1) >= self.wr_ob))  # crossunder(wr, -20)
        trend_exit = st_flip_down | wr_overbought

        c = close.to_numpy(); a = atr.to_numpy()
        buy = buy_cond.to_numpy(); texit = trend_exit.to_numpy()
        n = len(close); state = np.zeros(n, dtype=bool)
        on = False; stop = np.nan
        for i in range(n):
            if on:
                # 청산: 추세반전/%R 과매수 OR 고정 ATR 손절 이탈
                if texit[i] or (not np.isnan(stop) and c[i] <= stop):
                    on = False; stop = np.nan
            elif buy[i]:
                on = True
                # 진입 시점 ATR 을 캡처해 진입가 기준 고정 손절선 설정
                stop = c[i] - a[i] * self.atr_stop_mult if not np.isnan(a[i]) else np.nan
            state[i] = on

        target = pd.Series(state, index=close.index, name="target_long")
        return Signals(target_long=target,
                       indicators={f"Williams%R{self.wr_len}": wr},
                       overlays={f"EMA{self.ema_len}": ema200,
                                 "SuperTrend": st_line})
