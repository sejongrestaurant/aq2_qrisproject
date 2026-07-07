"""20SMA 기울기 + ROC 스윙 전략 (롱-플랫).

진입: 20SMA 기울기 > 0  **그리고**  ROC ≥ roc_th(%).
청산: 20SMA 기울기 ≤ 0 (추세선이 꺾이면 이탈).

기울기는 SMA 의 차분(sma - sma.shift(slope_len))으로, ROC 는 종가 변화율((close/close[-roc_len]-1)×100)로
계산한다. 진입은 두 조건 동시 충족, 청산은 기울기 조건만으로(ROC 무관) — "오를 때 타고 꺾이면 내린다".

신호는 봉 i 종가 기준 확정, 체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signals, Strategy


class SMASlopeROCStrategy(Strategy):
    """20SMA 기울기 + ROC 진입 / 기울기 반전 청산 롱-플랫 전략.

    Args (생성자):
        sma_len: 이동평균 기간(기본 20).
        slope_len: 기울기 차분 간격(기본 1 = 전봉 대비).
        roc_len: ROC 기간(기본 10). 종가 대비 며칠 전 대비 변화율인지.
        roc_th: 진입 요구 최소 ROC %(기본 0.5).
    """

    def __init__(
        self,
        sma_len: int = 20,
        slope_len: int = 1,
        roc_len: int = 10,
        roc_th: float = 0.5,
        roc_smooth: int | None = None,
        roc_slope_len: int | None = None,
        roc_floor: float | None = None,
        slope_enter_th: float = 0.0,
        slope_exit_th: float = 0.0,
        sma_dev_exit: float | None = None,
        atr_period: int = 14,
        atr_trailing: float | None = None,
        atr_stop_loss: float | None = None,
        name: str | None = None,
    ):
        sm = f"~{roc_smooth}" if roc_smooth else ""
        kind = f"ROC{roc_len}{sm}'slope" if roc_slope_len else f"ROC{roc_len}{sm}"
        fl = f"&ROC≥{roc_floor:g}" if roc_floor is not None else ""
        rg = f" [국면 진입≥{slope_enter_th:g}/청산≤{slope_exit_th:g}%]" \
            if (slope_enter_th or slope_exit_th) else ""
        dv = f"+dev{sma_dev_exit*100:g}%" if sma_dev_exit is not None else ""
        parts = ([f"SL{atr_stop_loss:g}"] if atr_stop_loss is not None else []) + \
                ([f"TS{atr_trailing:g}"] if atr_trailing is not None else [])
        st = "+ATR(" + "/".join(parts) + ")" if parts else ""
        super().__init__(name or f"SMA{sma_len}slope+{kind}≥{roc_th:g}{fl}{dv}{st}{rg}")
        self.sma_len = sma_len
        self.slope_len = slope_len
        self.roc_len = roc_len
        self.roc_th = roc_th
        self.roc_smooth = roc_smooth       # ROC 를 최근 N봉 평균으로 스무딩(None=원시)
        self.roc_slope_len = roc_slope_len  # 설정 시 ROC '기울기'(diff)로 진입 판정(모멘텀 가속)
        self.roc_floor = roc_floor          # 설정 시 ROC 레벨 하한 AND 조건(모멘텀 자체가 양수여야)
        # 국면 각도(정규화 기울기 %/봉) 문턱: 진입은 이 이상(상승국면), 청산은 이 이하(횡보/하락 진입)
        self.slope_enter_th = slope_enter_th
        self.slope_exit_th = slope_exit_th
        self.sma_dev_exit = sma_dev_exit    # 청산 OR 조건: 종가가 SMA 대비 이 비율 이상 하락(0.03=3%)
        self.atr_period = atr_period
        self.atr_trailing = atr_trailing    # 추적 손절 ATR 배수(최고 종가 기준, None=미사용)
        self.atr_stop_loss = atr_stop_loss  # 하드 손절 ATR 배수(진입가 기준, None=미사용)
        self._use_stops = atr_trailing is not None or atr_stop_loss is not None

    def _atr(self, data: pd.DataFrame) -> pd.Series:
        """Wilder ATR(손절 거리 계산용)."""
        high = data["high"] if "high" in data.columns else data["High"]
        low = data["low"] if "low" in data.columns else data["Low"]
        close = data["close"] if "close" in data.columns else data["Close"]
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close = data["close"] if "close" in data.columns else data["Close"]
        sma = close.rolling(self.sma_len).mean()
        # '각도' = 정규화 기울기(SMA 의 %/봉 변화율). 종목 스케일 무관하게 국면 문턱을 걸 수 있다.
        slope_pct = (sma - sma.shift(self.slope_len)) / sma * 100.0
        roc = (close / close.shift(self.roc_len) - 1.0) * 100.0
        if self.roc_smooth:                            # ROC 최근 N봉 평균 스무딩
            roc = roc.rolling(self.roc_smooth).mean()
        roc_signal = roc.diff(self.roc_slope_len) if self.roc_slope_len else roc

        # 진입: 상승국면(각도>진입문턱) AND ROC 가속/레벨
        long_ok = (slope_pct > self.slope_enter_th) & (roc_signal >= self.roc_th)
        if self.roc_floor is not None:                 # ROC 레벨 하한 추가(모멘텀 자체가 양수)
            long_ok = long_ok & (roc >= self.roc_floor)
        exit_ok = slope_pct <= self.slope_exit_th      # 청산: 각도가 청산문턱 이하(횡보/하락 진입)
        if self.sma_dev_exit is not None:              # OR 급락 방어: 종가가 SMA 대비 N% 이상 하락
            exit_ok = exit_ok | (close < sma * (1 - self.sma_dev_exit))

        lo = long_ok.to_numpy(); ex = exit_ok.to_numpy(); c = close.to_numpy()
        atr = self._atr(data) if self._use_stops else None
        atr_v = atr.to_numpy() if atr is not None else None
        n = len(close)
        state = np.zeros(n, dtype=bool)
        stop = np.full(n, np.nan)
        on = False
        entry_px = atr_entry = hwm = 0.0
        for i in range(n):
            if on:
                hwm = max(hwm, c[i])
                level = -np.inf                        # 유효 손절 레벨(하드·추적 중 더 타이트한 쪽)
                if self.atr_stop_loss is not None and not np.isnan(atr_entry):
                    level = max(level, entry_px - self.atr_stop_loss * atr_entry)
                if self.atr_trailing is not None and atr_v is not None and not np.isnan(atr_v[i]):
                    level = max(level, hwm - self.atr_trailing * atr_v[i])
                stop_hit = level > -np.inf and c[i] <= level
                if ex[i] or stop_hit:                  # 기울기 꺾임 또는 ATR 손절
                    on = False
                elif level > -np.inf:
                    stop[i] = level
            elif lo[i]:
                on = True
                entry_px = c[i]
                atr_entry = atr_v[i] if atr_v is not None else np.nan
                hwm = entry_px
            state[i] = on

        target = pd.Series(state, index=close.index, name="target_long")
        overlays = {f"SMA{self.sma_len}": sma}
        if self._use_stops:
            overlays["ATR Stop"] = pd.Series(stop, index=close.index, name="ATR Stop")
        return Signals(target_long=target,
                       indicators={f"ROC{self.roc_len}": roc},
                       overlays=overlays)
