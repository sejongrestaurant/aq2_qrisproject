"""일봉 TrendScore 기반 스윙 전략 (롱-플랫, 히스테리시스 + ADX 게이트 + 선택적 ATR 손절).

참고 트리의 스윙 자식(SwingExecutionStrategy)이 위성 ETF 를 일봉 TrendScore 로 진입/청산하는 로직을
재구성하고, 리스크 관리 장치를 옵션으로 얹는다:

  1. 히스테리시스(이중 임계): score ≥ ``entry`` 진입, score < ``exit`` 청산, 그 사이 상태 유지.
  2. ADX 게이트(선택): 진입 시 상승방향 ADX ≥ ``adx_gate`` 를 추가로 요구(추세 강도 확인).
  3. ATR 손절(선택): 아래 두 스탑이 TrendScore 청산선보다 **먼저** 포지션을 끊어 낙폭을 제한한다.
     · 하드 손절(stop-loss): 진입가 − ``atr_stop_loss`` × ATR(진입 시점). 최초 리스크 상한.
     · 추적 손절(trailing): 진입 후 최고 종가 − ``atr_trailing`` × ATR(현재). 이익을 따라 올라감.

라이브 규칙("TrendScore 36 돌파 + ADX≥28 매수 / 36 하회 매도")은 entry=exit=36, adx_gate=28 로 표현된다.
신호는 봉 i 종가 기준 확정, 체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicator import ADXIndicator, TrendScoreIndicator

from .base import Signals, Strategy


class TrendScoreSwingStrategy(Strategy):
    """TrendScore 롱-플랫 스윙 전략(히스테리시스 + ADX 게이트 + 선택적 ATR 손절).

    Args (생성자):
        entry / exit: 진입/청산 임계.
        adx_gate: 진입 시 요구 최소 ADX. None 이면 게이트 없음.
        adx_directional: True=상승방향 ADX≥gate, False=|ADX|≥gate.
        adx_period: ADX 기간(게이트용).
        atr_period: ATR 기간(손절용).
        atr_stop_loss: 하드 손절 ATR 배수(None=미사용).
        atr_trailing: 추적 손절 ATR 배수(None=미사용).
        indicator: 주입할 TrendScore 지표(없으면 기본 생성).

    Note:
        게이트가 없으면 entry > exit 이어야 히스테리시스 밴드가 성립한다(게이트가 있으면 entry==exit 허용).
        손절은 진입가 대비 종가 기준으로 판정한다(신호 레벨 근사). 실제 체결은 엔진이 익일 시가로 처리.
    """

    def __init__(
        self,
        entry: float = 60.0,
        exit: float = 45.0,
        adx_gate: float | None = None,
        adx_directional: bool = True,
        adx_period: int = 14,
        adx_dm_mode: str = "hybrid_max",   # max(highlow, close) DM: 종가 가속 얹되 고저 신호 유지(전종목 견고)
        adx_tr_body_alpha: float = 1.0,
        atr_period: int = 14,
        atr_stop_loss: float | None = None,
        atr_trailing: float | None = None,
        indicator: TrendScoreIndicator | None = None,
        name: str | None = None,
    ):
        if adx_gate is None and entry <= exit:
            raise ValueError(f"entry({entry})는 exit({exit})보다 커야 히스테리시스가 성립"
                             " (ADX 게이트가 없을 때)")
        if entry < exit:
            raise ValueError(f"entry({entry})는 exit({exit}) 이상이어야 함")

        self.entry = entry
        self.exit = exit
        self.adx_gate = adx_gate
        self.adx_directional = adx_directional
        self.atr_period = atr_period
        self.atr_stop_loss = atr_stop_loss
        self.atr_trailing = atr_trailing
        self.indicator = indicator or TrendScoreIndicator()
        self._adx = (ADXIndicator(adx_period, dm_mode=adx_dm_mode, tr_body_alpha=adx_tr_body_alpha)
                     if adx_gate is not None else None)
        self._use_stops = atr_stop_loss is not None or atr_trailing is not None

        super().__init__(name or self._default_name())

    def _default_name(self) -> str:
        name = f"TrendScoreSwing {self.entry:.0f}/{self.exit:.0f}"
        if self.adx_gate is not None:
            name += f"+ADX{self.adx_gate:.0f}"
        if self._use_stops:
            parts = []
            if self.atr_stop_loss is not None:
                parts.append(f"SL{self.atr_stop_loss:g}")
            if self.atr_trailing is not None:
                parts.append(f"TS{self.atr_trailing:g}")
            name += "+ATR(" + "/".join(parts) + ")"
        return name

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        score = self.indicator.compute(data)
        inds = {self.indicator.name: score}

        adx = None
        if self._adx is not None:
            adx_signed = self._adx.compute(data)
            adx = adx_signed if self.adx_directional else adx_signed.abs()
            inds[self._adx.name] = adx_signed

        atr = self._atr(data) if self._use_stops else None
        target, stop_line = self._state_machine(
            score, self.entry, self.exit, adx, self.adx_gate, data.get("close"), atr)

        overlays = {}
        if stop_line is not None:
            overlays["ATR Stop"] = stop_line
        return Signals(target_long=target, indicators=inds, overlays=overlays)

    def _atr(self, data: pd.DataFrame) -> pd.Series:
        """Wilder ATR(손절 거리 계산용)."""
        high = self.indicator._series(self.indicator._col(data, "high", "High"))
        low = self.indicator._series(self.indicator._col(data, "low", "Low"))
        close = self.indicator._series(self.indicator._col(data, "close", "Close", "adj_close"))
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()

    def _state_machine(self, score, entry, exit, adx, adx_gate, close, atr):
        """롱-플랫 상태기: 진입(score≥entry [+ADX게이트]) / 청산(score<exit 또는 ATR 손절).

        Returns:
            (target_long: pd.Series[bool], stop_line: pd.Series|None).
            stop_line 은 보유 중 유효 손절 레벨(하드·추적 중 높은 값), 플랫 구간은 NaN.
        """
        s = score.to_numpy()
        a = adx.to_numpy() if adx is not None else None
        c = close.to_numpy() if close is not None else None
        atr_v = atr.to_numpy() if atr is not None else None
        n = len(s)
        state = np.zeros(n, dtype=bool)
        stop = np.full(n, np.nan)

        on = False
        entry_px = atr_entry = hwm = 0.0
        for i in range(n):
            v = s[i]
            if np.isnan(v):
                on = False
                continue

            if on:
                hwm = max(hwm, c[i]) if c is not None else hwm
                # 유효 손절 레벨(하드·추적 중 더 높은 = 더 타이트한 쪽)
                level = -np.inf
                if self.atr_stop_loss is not None and not np.isnan(atr_entry):
                    level = max(level, entry_px - self.atr_stop_loss * atr_entry)
                if self.atr_trailing is not None and atr_v is not None and not np.isnan(atr_v[i]):
                    level = max(level, hwm - self.atr_trailing * atr_v[i])
                stop_hit = level > -np.inf and c is not None and c[i] <= level
                if v < exit or stop_hit:                    # 정상 청산 또는 손절
                    on = False
                else:
                    stop[i] = level if level > -np.inf else np.nan
            else:
                gate_ok = True
                if adx_gate is not None:
                    gate_ok = (a is not None and not np.isnan(a[i]) and a[i] >= adx_gate)
                if v >= entry and gate_ok:
                    on = True
                    entry_px = c[i] if c is not None else 0.0
                    atr_entry = atr_v[i] if atr_v is not None else np.nan
                    hwm = entry_px
                    if atr_v is not None:                   # 진입 봉 손절 레벨 표시
                        lvl = -np.inf
                        if self.atr_stop_loss is not None and not np.isnan(atr_entry):
                            lvl = max(lvl, entry_px - self.atr_stop_loss * atr_entry)
                        if self.atr_trailing is not None and not np.isnan(atr_v[i]):
                            lvl = max(lvl, hwm - self.atr_trailing * atr_v[i])
                        stop[i] = lvl if lvl > -np.inf else np.nan
            state[i] = on

        target = pd.Series(state, index=score.index, name="target_long")
        stop_line = (pd.Series(stop, index=score.index, name="ATR Stop")
                     if self._use_stops else None)
        return target, stop_line
