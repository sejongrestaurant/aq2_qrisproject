"""사테라이트(모멘텀 로테이션) 백테스터.

후보 유니버스의 각 종목에 지표(TrendScore)를 계산해 매 체크주기마다 **점수 상위 top_n 종목을
동일가중으로 보유**하고, 다음 체크에서 상위 구성이 바뀌면 교체한다. 룩어헤드를 막기 위해 종가
기준 점수로 선정한 목표를 **다음 거래일에 반영**하고, 종목 교체 시 회전율에 비례해 비용을 뺀다.

상장 시점이 다른 종목이 섞여도, 각 종목은 가격·점수가 유효한 날에만 선정 후보가 된다(상장 전·
지표 워밍업 구간은 자동 제외). 산출물은 기존 `BacktestResult` 로 포장해 리포트를 재사용한다.
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

from .config import SatelliteConfig

logger = logging.getLogger(__name__)


class SatelliteBacktester:
    """지표 점수 상위 top_n 동일가중 로테이션 백테스터.

    Args (생성자):
        loader: 종목 시세를 표준 스키마로 읽는 `DataLoader`.
        indicator: 순위 산정용 지표(예: `TrendScoreIndicator`). 각 종목 종가 시계열에 적용.
        cost: 왕복 거래비용 비율(예 0.0010). 종목 교체 회전율에 비례해 차감.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010):
        self.loader = loader
        self.indicator = indicator
        self.cost = cost

    # ── public ──────────────────────────────────────────────────────
    def run(self, scfg: SatelliteConfig, start=None, end=None) -> BacktestResult:
        """사테라이트 로테이션을 백테스트해 `BacktestResult` 로 반환한다.

        Args:
            scfg: 사테라이트 설정(유니버스·top_n·체크주기).
            start / end: 백테스트 구간(None 이면 미제한).
        Returns:
            자산곡선·벤치마크를 담은 `BacktestResult`(code="SATELLITE").
        """
        closes, scores = self._load_matrix(scfg.universe, start, end)
        equity, rotations, last_pick = self._simulate(closes, scores, scfg.top_n, scfg.check_period)
        benchmark = self._equal_weight_all(closes)

        logger.info(f"사테라이트 시뮬레이션 · 후보 {closes.shape[1]}종목 · "
                    f"{closes.index[0]:%Y-%m-%d}~{closes.index[-1]:%Y-%m-%d} · "
                    f"교체 {len(rotations)}회 · 최근 보유 {last_pick}")
        return self._to_result(scfg, closes, equity, benchmark, rotations)

    # ── 데이터 준비 ─────────────────────────────────────────────────
    def _load_matrix(self, universe: List[str], start, end) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """유니버스의 종가·지표점수 행렬을 만든다(로드 실패 종목은 제외).

        날짜는 모든 후보의 합집합으로 두고(상장 전은 NaN), 지표는 각 종목 종가에 계산 후 같은
        날짜축에 정렬한다. 이렇게 하면 상장이 늦은 종목도 유효해지는 날부터 선정 후보가 된다.
        """
        close_cols: Dict[str, pd.Series] = {}
        score_cols: Dict[str, pd.Series] = {}
        for code in universe:
            try:
                price = self.loader.load(code)
            except Exception as exc:  # noqa: BLE001 — 개별 종목 실패가 전체를 막지 않도록
                logger.warning(f"{code}: 로드 실패 → 후보에서 제외 ({exc})")
                continue
            close_cols[code] = price.df["close"]
            score_cols[code] = self.indicator.compute(price.df)
        if not close_cols:
            raise RuntimeError("사테라이트: 로드 가능한 후보 종목이 없습니다.")

        closes = pd.DataFrame(close_cols).sort_index().loc[start:end]
        # 모든 후보가 아직 없던 초기 구간(전 종목 NaN 행)은 버린다.
        closes = closes.dropna(how="all")
        scores = pd.DataFrame(score_cols).reindex(index=closes.index, columns=closes.columns)
        if len(closes) < 2:
            raise ValueError("사테라이트: 백테스트할 거래일이 부족합니다.")
        return closes, scores

    # ── 시뮬레이션 ──────────────────────────────────────────────────
    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame, top_n: int,
                  period: str) -> Tuple[pd.Series, List[pd.Timestamp], List[str]]:
        """점수 상위 top_n 동일가중 로테이션 시뮬레이션.

        각 체크일 종가 점수로 상위 종목을 뽑아 **다음 거래일**에 목표비중(동일가중)으로 재조정한다.
        보유 구성(집합)이 바뀐 날을 교체(rotation)로 기록한다.

        Returns:
            (equity 시작 1.0, rotations 교체일 리스트, last_pick 마지막 보유 종목명 리스트).
        """
        idx = closes.index
        tickers = list(closes.columns)
        C = closes.to_numpy(dtype=float)
        R = np.zeros_like(C)
        R[1:] = C[1:] / C[:-1] - 1.0                 # 일간수익(가격 결측 구간은 NaN)
        S = scores.to_numpy(dtype=float)             # 점수(워밍업·상장전 NaN)
        T, N = C.shape
        check = period_mask(idx, period)

        value = np.zeros(N)      # 종목별 보유 가치
        holding = False
        pending = None           # 다음 거래일에 반영할 목표비중
        eq = np.empty(T)
        rotations: List[pd.Timestamp] = []
        prev_set: frozenset = frozenset()

        for i in range(T):
            # (a) 당일 수익 반영(첫 투자 이후). 결측 수익은 0 으로 간주.
            if holding:
                value = value * (1.0 + np.nan_to_num(R[i], nan=0.0))
            # (b) 전 체크에서 정한 목표를 오늘 반영(교체/재조정 + 회전율 비용)
            if pending is not None:
                total = value.sum() if holding else 1.0
                w_now = (value / total) if (holding and total > 0) else np.zeros(N)
                turnover = 0.5 * np.abs(pending - w_now).sum()
                total *= (1.0 - self.cost * turnover)
                value = pending * total
                holding = True
                pending = None
            eq[i] = value.sum() if holding else 1.0
            # (c) 오늘 종가 점수로 상위 top_n 선정 → 다음 거래일 목표로 예약
            if check[i]:
                row = S[i]
                valid = np.where(~np.isnan(row) & ~np.isnan(C[i]))[0]  # 점수·가격 유효 종목만
                if len(valid) >= 1:
                    ranked = valid[np.argsort(row[valid])[::-1]]       # 점수 내림차순
                    sel = ranked[:top_n]
                    w_t = np.zeros(N)
                    w_t[sel] = 1.0 / len(sel)
                    pending = w_t
                    cur_set = frozenset(int(s) for s in sel)
                    if cur_set != prev_set:                            # 보유 구성 변경 = 교체
                        rotations.append(idx[i])
                        prev_set = cur_set

        last_pick = [tickers[j] for j in sorted(prev_set)] if prev_set else []
        return pd.Series(eq, index=idx, name="equity"), rotations, last_pick

    @staticmethod
    def _equal_weight_all(closes: pd.DataFrame) -> pd.Series:
        """벤치마크: 후보 전 종목을 매일 동일가중 보유(선정 없이 '전부 보유')한 자산곡선."""
        daily = closes.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
        return (1.0 + daily).cumprod().rename("buy_and_hold")

    # ── 결과 포장 ───────────────────────────────────────────────────
    def _to_result(self, scfg: SatelliteConfig, closes: pd.DataFrame, equity: pd.Series,
                   benchmark: pd.Series, rotations: List[pd.Timestamp]) -> BacktestResult:
        """자산곡선을 기존 `BacktestResult` 로 감싼다(리포트·지표 재사용).

        · price: 가격 패널에 '전 종목 동일가중(벤치마크)' 곡선을 실어 선정 효과를 대비.
        · target_long: 항상 전액 투자이므로 전 구간 True.
        · trades: 종목 교체 구간별 보유거래(교체 활동을 리포트 거래 테이블로 표현).
        """
        bench_vals = benchmark.to_numpy()
        price_df = pd.DataFrame(
            {"open": bench_vals, "high": bench_vals, "low": bench_vals, "close": bench_vals},
            index=closes.index)
        target_long = pd.Series(True, index=closes.index)

        return BacktestResult(
            code="SATELLITE",
            strategy_name=f"Top{scfg.top_n} {self.indicator.name} {scfg.check_period}체크",
            name=scfg.name,
            equity=equity,
            benchmark=benchmark,
            trades=segment_trades(equity, rotations, reason="교체"),
            price=price_df,
            target_long=target_long,
            indicators={},
            overlays={},
            cost=self.cost,
        )
