"""IRP(개인형 퇴직연금) ETF 전략 백테스터.

**채권 30% 고정 + 사테라이트 70%** 를 분기마다 목표비중으로 되돌리는 자산배분을 시뮬레이션한다.
핵심은 합성(composition)이다: 70% 사테라이트 슬리브는 기존 `SatelliteBacktester` 로 돌려 얻은
자산곡선을 **하나의 합성 자산**(일간수익 스트림)으로 취급하고, 채권 3종과 함께 4-슬리브 포트폴리오로
묶어 분기 리밸런싱한다. 이렇게 하면 월간 로테이션(슬리브 내부)과 분기 리밸런싱(슬리브 간)이 자연히
2단으로 나뉜다(관심사 분리).

체결 규약: 일간 종가 기준 평가·리밸런싱(결정과 실행이 같은 종가 → 룩어헤드 없음). 사테라이트
슬리브의 로테이션 비용은 슬리브 곡선에 이미 반영돼 있고, 분기 리밸런싱 회전율 비용만 추가로 뺀다.

산출물은 기존 `BacktestResult` 로 포장해 리포트·성과지표 계층을 그대로 재사용한다.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest import BacktestResult
from data import DataLoader
from indicator import Indicator
from portfolio.schedule import period_mask, segment_trades
from satellite import SatelliteBacktester

from .config import IRPConfig

logger = logging.getLogger(__name__)

# 사테라이트 슬리브를 4-슬리브 포트폴리오에서 가리키는 합성 자산 키(실제 티커와 겹치지 않게).
_SAT_KEY = "__SAT__"


class IRPBacktester:
    """채권 고정 + 사테라이트 로테이션 자산배분 백테스터.

    Args (생성자):
        loader: 종목 시세를 표준 스키마로 읽는 `DataLoader`.
        indicator: 사테라이트 순위 산정 지표(예: `TrendScoreIndicator`).
        cost: 왕복 거래비용 비율(예 0.0010). 사테라이트 내부 로테이션·분기 리밸런싱에 공통 적용.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010):
        self.loader = loader
        self.cost = cost
        self.satellite = SatelliteBacktester(loader=loader, indicator=indicator, cost=cost)

    # ── public ──────────────────────────────────────────────────────
    def run(self, icfg: IRPConfig, start=None, end=None) -> BacktestResult:
        """IRP 전략을 백테스트해 `BacktestResult` 로 반환한다(code="IRP").

        Args:
            icfg: IRP 설정(채권 비중·사테라이트 슬리브·리밸런싱 주기).
            start / end: 백테스트 구간(None 이면 미제한).
        """
        # (1) 70% 사테라이트 슬리브: 트레일링 스탑 없이 순수 Top-N 월간 로테이션.
        sat = self.satellite.run(icfg.satellite, start=start, end=end, trailing=None)
        # (2) 슬리브 일간수익 + 채권 일간수익을 공통 거래일에 정렬(4-슬리브 수익 행렬).
        rets = self._combine_returns(sat.equity, icfg.bonds, start, end)
        weights = {_SAT_KEY: icfg.satellite_weight, **icfg.bonds}
        # (3) 분기 리밸런싱 시뮬 + 무리밸런싱(드리프트) 벤치마크.
        equity, rb_dates = self._simulate(rets, weights, icfg.rebalance_period)
        benchmark = self._buy_and_hold(rets, weights)

        logger.info(f"IRP 시뮬레이션 · 채권 {len(icfg.bonds)}종 {icfg.bond_weight * 100:.0f}% "
                    f"+ 사테라이트 {icfg.satellite_weight * 100:.0f}% · "
                    f"{rets.index[0]:%Y-%m-%d}~{rets.index[-1]:%Y-%m-%d} · "
                    f"분기 리밸런싱 {len(rb_dates)}회")
        # 사테라이트 슬리브의 월간 선정 이력을 IRP 결과에 실어 리포트에 로테이션 내역을 보여준다.
        return self._to_result(icfg, rets.index, equity, benchmark, rb_dates, sat.rotations_log)

    # ── 데이터 준비 ─────────────────────────────────────────────────
    def _combine_returns(self, sat_equity: pd.Series, bonds: Dict[str, float],
                         start, end) -> pd.DataFrame:
        """사테라이트 슬리브 수익 + 채권 수익을 공통 거래일에 정렬한 수익 행렬을 만든다.

        모든 슬리브가 값을 가진 날만 남긴다(dropna). 채권 상장이 늦으면 그 이후부터 백테스트된다.
        슬리브·채권 모두 한국거래소 달력이라 거래일이 일치한다(합성 자산과 채권 정렬이 깔끔).
        """
        bond_close: Dict[str, pd.Series] = {}
        for code in bonds:
            try:
                bond_close[code] = self.loader.load(code).df["close"]
            except Exception as exc:  # noqa: BLE001 — 채권 로드 실패는 전체를 막으므로 치명 처리
                raise RuntimeError(f"IRP: 채권 {code} 로드 실패 ({exc})") from exc

        cols = {_SAT_KEY: sat_equity}
        cols.update(bond_close)
        panel = pd.DataFrame(cols).sort_index().loc[start:end].dropna()
        if len(panel) < 2:
            raise ValueError("IRP: 사테라이트·채권 공통 거래일이 부족합니다(구간이 겹치지 않음).")
        # 합성 자산(슬리브 곡선)·채권 종가를 각각 일간수익으로 환산(첫날 0).
        return panel.pct_change(fill_method=None).fillna(0.0)

    # ── 시뮬레이션 ──────────────────────────────────────────────────
    def _simulate(self, rets: pd.DataFrame, weights: Dict[str, float],
                  period: str) -> Tuple[pd.Series, List[pd.Timestamp]]:
        """4-슬리브 비중 합성 자산곡선 + 분기 리밸런싱 시뮬레이션.

        각 슬리브 가치를 일간수익으로 굴리고, 리밸런싱 주기의 첫 거래일에 목표비중(채권 10/10/10 +
        사테라이트 70)으로 되돌린다. 리밸런싱 회전율(단방향)에 비례해 왕복비용을 뺀다.

        Returns:
            (equity: 시작 1.0 자산곡선, rb_dates: 실제 리밸런싱이 일어난 날짜 리스트).
        """
        cols = list(rets.columns)
        w_t = np.array([weights[c] for c in cols], dtype=float)  # 목표비중(합=1.0)
        R = rets.to_numpy()
        dates = rets.index
        n = len(dates)
        periodic = period_mask(dates, period)

        value = w_t.copy()  # 슬리브별 가치(합=1.0)
        eq = np.empty(n)
        rb_dates: List[pd.Timestamp] = []
        for i in range(n):
            if i > 0:
                value = value * (1.0 + R[i])   # 당일 수익 반영
            total = value.sum()
            if i > 0 and periodic[i]:           # 분기 첫 거래일 → 목표비중 복원
                w_now = value / total
                turnover = 0.5 * np.abs(w_now - w_t).sum()
                total *= (1.0 - self.cost * turnover)
                value = w_t * total
                rb_dates.append(dates[i])
            eq[i] = total
        return pd.Series(eq, index=dates, name="equity"), rb_dates

    @staticmethod
    def _buy_and_hold(rets: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """벤치마크: 초기 비중(30채권/70사테라이트)으로 사서 **분기 리밸런싱 없이** 굴린 곡선.

        리밸런싱의 순효과가 전략 곡선 대 이 벤치마크 차이로 드러난다(포트폴리오 계층과 동일 규약).
        """
        cols = list(rets.columns)
        w = np.array([weights[c] for c in cols], dtype=float)
        growth = (1.0 + rets).cumprod().to_numpy()   # 슬리브별 성장(각 시작 1.0)
        bh = (growth * w).sum(axis=1)
        return pd.Series(bh, index=rets.index, name="buy_and_hold")

    # ── 결과 포장 ───────────────────────────────────────────────────
    def _to_result(self, icfg: IRPConfig, index: pd.DatetimeIndex, equity: pd.Series,
                   benchmark: pd.Series, rb_dates: List[pd.Timestamp],
                   rotations_log) -> BacktestResult:
        """자산곡선을 기존 `BacktestResult` 로 감싼다(리포트·지표 재사용).

        · price: 가격 패널에 '무리밸런싱 드리프트(벤치마크)' 곡선을 실어 리밸런싱 효과를 대비.
        · target_long: 항상 전액 투자이므로 전 구간 True(노출 100%).
        · trades: 분기 리밸런싱 구간별 보유거래(리밸런싱 활동을 거래 테이블로 표현).
        · rotations_log: 70% 슬리브의 월간 섹터 선정 이력(리포트 로테이션 내역 표).
        """
        bench_vals = benchmark.to_numpy()
        price_df = pd.DataFrame(
            {"open": bench_vals, "high": bench_vals, "low": bench_vals, "close": bench_vals},
            index=index)
        target_long = pd.Series(True, index=index)

        s = icfg.satellite
        name = (f"채권{icfg.bond_weight * 100:.0f}/사테{icfg.satellite_weight * 100:.0f} · "
                f"분기리밸 · Top{s.top_n} {s.check_period}로테이션")
        return BacktestResult(
            code="IRP",
            strategy_name=name,
            name=icfg.name,
            equity=equity,
            benchmark=benchmark,
            trades=segment_trades(equity, rb_dates, reason="리밸런싱"),
            price=price_df,
            target_long=target_long,
            indicators={},
            overlays={},
            cost=self.cost,
            rotations_log=rotations_log,
        )
