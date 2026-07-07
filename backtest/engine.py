"""롱-플랫 백테스트 엔진.

전략의 목표 보유상태(봉 i 종가 확정)를 받아 **익일(i+1) 시가 체결** 로 자산곡선을 계산한다.
룩어헤드를 원천 차단하고(신호는 항상 체결보다 하루 앞섬), 왕복 거래비용을 청산 시 1회 차감한다.
숏 없는 전액 투입/전액 청산 모델로, 참고 트리 스윙 자식의 체결 규약과 동일하다.

확장: 목표 비중·부분 체결·복수 종목 포트폴리오가 필요하면 본 엔진을 상속하거나 자매 엔진을 추가하되,
전략/리포트 인터페이스(`Strategy` → `Signals`, `Backtester.run` → `BacktestResult`)는 유지한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import PriceData
from strategy import Strategy

from .result import BacktestResult
from .trade import Trade


class Backtester:
    """익일 시가 체결 롱-플랫 백테스터.

    Args (생성자):
        cost: 왕복 거래비용(수수료+슬리피지) 비율. 예 0.0010 = 0.10%. 청산 시 1회 차감.
    """

    def __init__(self, cost: float = 0.0010):
        self.cost = cost

    # ── public ──────────────────────────────────────────────────────
    def run(self, price: PriceData, strategy: Strategy,
            start=None, end=None) -> BacktestResult:
        """전략을 시세에 적용해 백테스트를 실행한다.

        지표(신호)는 **받은 시세 전체**(워밍업 포함)로 계산한 뒤, 실제 매매·성과 집계는
        ``[start:end]`` 구간으로 한정한다. 즉 start 이전 데이터는 지표 예열용으로만 쓰이고
        거래·자산곡선·지표차트는 사용자가 지정한 구간에서 시작한다(워밍업 자동 반영).

        Args:
            price: 표준 스키마 `PriceData`(워밍업 봉이 앞에 포함된 상태).
            strategy: `Strategy` 인스턴스.
            start / end: 백테스트 구간("YYYY-MM-DD"·Timestamp·None). None 이면 미제한.
        Returns:
            자산곡선·거래·지표를 담은 `BacktestResult`.
        """
        full_df = price.df
        signals = strategy.generate_signals(full_df)  # 전체 구간으로 지표 예열

        # 워밍업 이후의 매매 구간으로 슬라이스
        window = full_df.loc[start:end]
        if window.empty:
            raise ValueError(f"{price.code}: 백테스트 구간이 비어 있음(start={start}, end={end})")
        target = signals.target_long.loc[window.index]
        indicators = {k: v.loc[window.index] for k, v in signals.indicators.items()}
        overlays = {k: v.loc[window.index] for k, v in signals.overlays.items()}

        equity, trades = self._simulate(window, target)
        benchmark = self._buy_and_hold(window)
        return BacktestResult(
            code=price.code,
            strategy_name=strategy.name,
            name=price.name,
            equity=equity,
            benchmark=benchmark,
            trades=trades,
            price=window,
            target_long=target,
            indicators=indicators,
            overlays=overlays,
            cost=self.cost,
        )

    # ── 체결 시뮬레이션 ──────────────────────────────────────────────
    def _simulate(self, df: pd.DataFrame, target_long: pd.Series):
        """봉 i 목표상태 → 봉 i+1 시가 체결 롱-플랫 시뮬레이션.

        Returns:
            (equity: pd.Series 시작 1.0, trades: List[Trade]).
        """
        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        want = target_long.reindex(df.index).fillna(False).to_numpy()
        n = len(df)
        idx = df.index

        position = 0            # 0=현금, 1=롱
        entry_px = 0.0
        entry_i = -1
        equity = 1.0
        eq_curve = np.full(n, np.nan)
        trades: list[Trade] = []

        for i in range(n - 1):
            exec_px = opens[i + 1]                       # 익일 시가 체결가
            go_long = bool(want[i]) and not np.isnan(exec_px)

            if position == 0 and go_long:
                position, entry_px, entry_i = 1, exec_px, i + 1
            elif position == 1 and not go_long:
                equity, trade = self._close(
                    equity, entry_px, exec_px, idx[entry_i], idx[i + 1],
                    (i + 1) - entry_i, "signal")
                trades.append(trade)
                position, entry_px, entry_i = 0, 0.0, -1

            # 일별 시가평가(mark-to-market): 보유 중이면 미실현손익 반영
            if position == 1:
                eq_curve[i + 1] = equity * (opens[i + 1] / entry_px) * (1 - self.cost)
            else:
                eq_curve[i + 1] = equity

        # 마지막 봉에 미청산 포지션이 남아 있으면 최종 종가로 강제 청산(보수적)
        if position == 1:
            equity, trade = self._close(
                equity, entry_px, closes[-1], idx[entry_i], idx[-1],
                (n - 1) - entry_i, "eod")
            trades.append(trade)
            eq_curve[-1] = equity

        eq = pd.Series(eq_curve, index=idx).ffill().fillna(1.0)
        return eq.rename("equity"), trades

    def _close(self, equity, entry_px, exit_px, entry_date, exit_date, bars, reason):
        """포지션 청산 회계: 왕복 비용 1회 차감한 순수익을 자산에 곱하고 Trade 를 만든다."""
        net = (exit_px / entry_px) * (1 - self.cost)
        equity *= net
        trade = Trade(
            entry_date=entry_date, exit_date=exit_date,
            entry_px=float(entry_px), exit_px=float(exit_px),
            ret=float(net - 1.0), bars_held=int(bars), exit_reason=reason)
        return equity, trade

    @staticmethod
    def _buy_and_hold(df: pd.DataFrame) -> pd.Series:
        """비교 벤치마크: 첫 종가 매수 후 보유하는 자산곡선(시작 1.0)."""
        close = df["close"]
        return (close / close.iloc[0]).rename("buy_and_hold")
