"""레짐 게이트 TrendScore 스윙 전략 (진입=TrendScore, 청산=수급배신 lifeline).

두 접근의 장점만 합성한다:

  · **진입/레짐** = TrendScore 점수 매매법 + ADX28 게이트.
  · **청산/리스크** = 수급배신 감지기 v5.3 의 lifeline(구조 스탑) + 구조붕괴 즉시청산.

배경(왜 이렇게 나누나):
  TrendScore 청산(score < exit)은 EWMAC span 128·TSMOM 252일 같은 긴 평활 성분의 합성이라
  급반전에 느리게 반응 → 낙폭이 깊어진다(MDD 열위). 반면 v5.3 lifeline 은 스윙저점을 올리기만 하는
  **가격 구조 스탑**이고 bearRun≥confirm_bear 구조붕괴가 빠른 2차 방어라, 실제 스윙이 깨지는 지점에서
  끊어 MDD 를 좁힌다. 그래서 "좋은 진입 판별(TrendScore) + 좋은 손실 컷(v5.3)" 으로 역할을 나눈다.

진입은 **이중 게이트**다(셋 다 동시 충족 시에만 진입):
  1. v5.3 bullState        — MA20/MA60 스프레드 ≥ spread_th 가 confirm_bull 봉 확인(히스테리시스).
  2. TrendScore ≥ ts_entry — 기본 50(WEAK_BULL) = "bull state 이상".
  3. 방향성 ADX ≥ adx_gate — 추세 강도(라이브 규칙 28).
  (+ MA240 맥락 필터, 쿨다운.)

청산은 v5.3 가 소유한다(TrendScore 하락으로는 청산하지 않음 — 그게 MDD 우위의 핵심):
  · lifeline 붕괴: close < lifeline − life_buf_atr×ATR.
  · 구조붕괴:    bearRun ≥ confirm_bear.
  · (옵션) ts_exit: score < ts_exit 백스톱. 기본 None(끔) — v5.3 청산과 controlled 비교용.

신호는 봉 i 종가 기준 확정, 체결은 backtest 엔진이 익일 시가로 처리(룩어헤드 방지).

주의(arm_phase2 기본 True): v5.3 원본은 lifeline 을 phase1(강추세, 스프레드≥spread_strong)에서만
  arming 하지만, 이 전략은 진입 게이트가 phase2(스프레드 2~5%)에서도 열릴 수 있다. phase1 만 arming 하면
  phase2 진입분에는 lifeline 이 안 잡혀 MDD 방어가 구조붕괴에만 의존하게 되므로, 여기서는 기본 True 로
  둬 phase2 에서도 lifeline 이 걸리게 한다(원본 대비 의도적 변경).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicator import ADXIndicator, TrendScoreIndicator

from .base import Signals, Strategy


class RegimeGatedTrendScoreStrategy(Strategy):
    """TrendScore 진입 + v5.3 lifeline 청산 롱-플랫 스윙 전략(이중 게이트).

    Args (생성자):
        ts_entry: 진입 요구 TrendScore(기본 50 = WEAK_BULL, "bull state 이상").
        ts_exit: score 백스톱 청산 임계(None=미사용, v5.3 청산이 소유).
        adx_gate: 진입 요구 최소 ADX(기본 28). None=게이트 없음.
        adx_directional: True=상승방향 ADX≥gate, False=|ADX|≥gate.
        ma_fast / ma_slow / ma_ctx: v5.3 구조 MA(20/60/240).
        spread_th / spread_strong: 약추세/강추세 스프레드 임계(%).
        exit_th: 구조 붕괴 판정 스프레드 임계(%).
        confirm_bull / confirm_bear: 상승/하락 전환 확인 봉수(히스테리시스).
        cooldown: 청산 후 재진입 금지 봉수.
        swing_n: lifeline 스윙저점 창.
        use_ma_ctx: True 면 close > MA240 진입 필터.
        arm_phase2: True 면 phase2(약추세대)에서도 lifeline arming(위 주의 참조).
        reset_regime_on_stop: True(기본)면 lifeline 붕괴 시 bullState 도 리셋(v5.3 동작) →
            재진입에 confirm_bull 봉 재확인 필요. False 면 regime 유지 → 쿨다운만 지나면
            (score·ADX 재충족 시) 즉시 재진입(grind-up 재진입 공백 보완).
        life_buf_atr: lifeline ATR 완충(청산선 = lifeline − k×ATR). 0=완충 없음.
        atr_period: 완충용 ATR 기간.
        indicator: 주입할 TrendScore 지표(없으면 기본 생성).

    Note:
        이중 게이트라 진입 빈도가 낮다(거래 수 적을 수 있음). v5.3 청산이 laggy score 청산을
        대체하므로 MDD 개선을 목표로 한다. 손절은 종가 기준 판정(신호 레벨 근사).
    """

    def __init__(
        self,
        ts_entry: float = 50.0,
        ts_exit: float | None = None,
        adx_gate: float | None = 28.0,
        adx_directional: bool = True,
        adx_period: int = 14,
        adx_dm_mode: str = "hybrid_max",   # max(highlow, close) DM: 종가 가속 얹되 고저 신호 유지(전종목 견고)
        adx_tr_body_alpha: float = 1.0,
        ma_fast: int = 20,
        ma_slow: int = 60,
        ma_ctx: int = 240,
        spread_th: float = 2.0,
        spread_strong: float = 5.0,
        exit_th: float = 0.0,
        confirm_bull: int = 3,
        confirm_bear: int = 3,
        cooldown: int = 10,
        swing_n: int = 10,
        use_ma_ctx: bool = True,
        arm_phase2: bool = True,
        reset_regime_on_stop: bool = True,
        life_buf_atr: float = 0.0,
        atr_period: int = 14,
        indicator: TrendScoreIndicator | None = None,
        name: str | None = None,
    ):
        if ts_exit is not None and ts_exit > ts_entry:
            raise ValueError(f"ts_exit({ts_exit})는 ts_entry({ts_entry}) 이하여야 함")
        if ma_fast >= ma_slow:
            raise ValueError(f"ma_fast({ma_fast})는 ma_slow({ma_slow})보다 작아야 함")

        self.ts_entry = ts_entry
        self.ts_exit = ts_exit
        self.adx_gate = adx_gate
        self.adx_directional = adx_directional
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_ctx = ma_ctx
        self.spread_th = spread_th
        self.spread_strong = spread_strong
        self.exit_th = exit_th
        self.confirm_bull = confirm_bull
        self.confirm_bear = confirm_bear
        self.cooldown = cooldown
        self.swing_n = swing_n
        self.use_ma_ctx = use_ma_ctx
        self.arm_phase2 = arm_phase2
        self.reset_regime_on_stop = reset_regime_on_stop
        self.life_buf_atr = life_buf_atr
        self.atr_period = atr_period
        self.indicator = indicator or TrendScoreIndicator()
        self._adx = (ADXIndicator(adx_period, dm_mode=adx_dm_mode, tr_body_alpha=adx_tr_body_alpha)
                     if adx_gate is not None else None)

        super().__init__(name or self._default_name())

    def _default_name(self) -> str:
        name = f"RegimeTS {self.ts_entry:.0f}"
        if self.adx_gate is not None:
            name += f"+ADX{self.adx_gate:.0f}"
        name += "+Lifeline"
        if self.life_buf_atr > 0:
            name += f"({self.life_buf_atr:g}ATR)"
        return name

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        ind = self.indicator
        close = ind._series(ind._col(data, "close", "Close", "adj_close"))
        if close is None:
            raise ValueError("RegimeGatedTrendScoreStrategy: 'close' 컬럼 필요")
        high = ind._col(data, "high", "High")
        low = ind._col(data, "low", "Low")
        high = ind._series(high if high is not None else close)
        low = ind._series(low if low is not None else close)

        score = ind.compute(data)
        ma_fast = close.rolling(self.ma_fast).mean()
        ma_slow = close.rolling(self.ma_slow).mean()
        ma_ctx = close.rolling(self.ma_ctx).mean()
        sspread = (ma_fast / ma_slow - 1.0) * 100.0
        swing_low = low.rolling(self.swing_n, min_periods=self.swing_n).min()
        atr = self._atr(high, low, close)

        adx = None
        inds = {ind.name: score}
        if self._adx is not None:
            adx_signed = self._adx.from_ohlc(high, low, close, self._adx.period)
            adx = adx_signed if self.adx_directional else adx_signed.abs()
            inds[self._adx.name] = adx_signed

        target, lifeline_line = self._state_machine(
            score, sspread, close, ma_ctx, swing_low, atr, adx)

        overlays = {"Lifeline": lifeline_line}
        return Signals(target_long=target, indicators=inds, overlays=overlays)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Wilder ATR(lifeline 완충 거리 계산용)."""
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(com=self.atr_period - 1, min_periods=self.atr_period).mean()

    def _state_machine(self, score, sspread, close, ma_ctx, swing_low, atr, adx):
        """진입=이중 게이트, 청산=lifeline·구조붕괴 롱-플랫 상태기.

        v5.3 봉 처리 순서를 이식한다: 쿨다운 감소 → run 카운터 → bullState 전환 →
        lifeline arming → 진입 → lifeline 청산. 진입·청산이 같은 봉에 겹치면 순 플랫(v5.3 동일).

        Returns:
            (target_long: pd.Series[bool], lifeline_line: pd.Series). lifeline_line 은
            bullState 구간의 lifeline 수준(플랫/미arming 구간은 NaN, 리포트 오버레이용).
        """
        s = score.to_numpy()
        sp = sspread.to_numpy()
        c = close.to_numpy()
        ctx = ma_ctx.to_numpy()
        sl = swing_low.to_numpy()
        av = atr.to_numpy()
        a = adx.to_numpy() if adx is not None else None
        n = len(s)

        state = np.zeros(n, dtype=bool)
        life_out = np.full(n, np.nan)

        on = False
        bull_state = False
        bull_run = bear_run = cool_left = 0
        lifeline = np.nan

        for i in range(n):
            cool_left = max(cool_left - 1, 0)

            ss = sp[i]
            entry_ok = not np.isnan(ss) and ss >= self.spread_th
            exit_ok = not np.isnan(ss) and ss < -self.exit_th
            bull_run = bull_run + 1 if (entry_ok and cool_left == 0) else 0
            bear_run = bear_run + 1 if exit_ok else 0

            # ── bullState 전환(히스테리시스) ──
            if bull_state and bear_run >= self.confirm_bear:   # 구조붕괴 → 청산
                bull_state = False
                on = False
                lifeline = np.nan
            if not bull_state and bull_run >= self.confirm_bull:
                bull_state = True
                lifeline = np.nan

            # ── lifeline arming(올리기만) ──
            arm_ok = (not np.isnan(ss)) and (
                ss >= self.spread_strong or (self.arm_phase2 and ss >= self.spread_th))
            if bull_state and arm_ok and not np.isnan(sl[i]):
                lifeline = sl[i] if np.isnan(lifeline) else max(lifeline, sl[i])

            # ── 진입: 이중 게이트(bullState AND TrendScore AND ADX) + 맥락·쿨다운 ──
            ctx_ok = (not self.use_ma_ctx) or np.isnan(ctx[i]) or c[i] > ctx[i]
            ts_bull = not np.isnan(s[i]) and s[i] >= self.ts_entry
            adx_ok = self.adx_gate is None or (
                a is not None and not np.isnan(a[i]) and a[i] >= self.adx_gate)
            if (not on) and bull_state and ts_bull and adx_ok and ctx_ok and cool_left == 0:
                on = True

            # ── 청산: v5.3 소유(lifeline 붕괴 / 옵션 score 백스톱) ──
            if on:
                life_stop = (np.nan if np.isnan(lifeline)
                             else lifeline - self.life_buf_atr * av[i])
                sell_trail = not np.isnan(life_stop) and c[i] < life_stop
                ts_break = self.ts_exit is not None and not np.isnan(s[i]) and s[i] < self.ts_exit
                if sell_trail or ts_break:
                    on = False
                    cool_left = self.cooldown
                    bull_run = 0
                    if sell_trail:            # lifeline 붕괴: lifeline 리셋(재arming), regime 은 옵션
                        lifeline = np.nan
                        if self.reset_regime_on_stop:   # v5.3 기본: regime 도 리셋(재확인 필요)
                            bull_state = False

            state[i] = on
            if bull_state and not np.isnan(lifeline):
                life_out[i] = lifeline

        target = pd.Series(state, index=score.index, name="target_long")
        lifeline_line = pd.Series(life_out, index=score.index, name="Lifeline")
        return target, lifeline_line
