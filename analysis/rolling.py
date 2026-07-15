"""롤링 시작점 분석 — "언제 시작했느냐" 가 결과를 얼마나 가르는가.

단일 구간 백테스트(2020-01 시작)의 성적은 **시작일 하나에 걸린 우연**이다. 하필 바닥에서
시작했으면 좋아 보이고 꼭지에서 시작했으면 나빠 보인다. IRP 가입자는 자기가 가입한 달에
시작할 뿐 시작일을 고를 수 없다. 그래서 제안서에 필요한 건 한 점의 수익률이 아니라
**모든 시작점의 분포**와 **보유기간별 손실 확률**이다.

한계(정직하게 명시할 것): 백테스트 구간이 2020-01~2026-06 = 78개월뿐이라, 보유기간이 길수록
표본 창이 급격히 줄고(60개월 보유 → 19창) 창끼리 구간이 겹쳐 독립 표본이 아니다. 즉 이
손실 확률은 '이 6.5년 안에서 시작 시점을 굴렸을 때' 의 값이지 미래 확률의 추정치가 아니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from .cashflow import CashflowPlan
from .dca import DCASimulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HorizonStats:
    """한 (납입계획, 보유기간) 조합의 롤링 통계.

    Attributes:
        plan_name: 납입 계획 표시명.
        horizon_months: 보유기간(개월).
        n_windows: 표본 창 수(겹치는 창이므로 독립 표본이 아니다).
        loss_prob_pct: 총 납입액 대비 손실(평가액 < 납입액)로 끝난 창의 비율(%).
        median_pct / worst_pct / best_pct: 납입액 대비 손익률(%)의 중앙값·최악·최선.
        median_mwr_pct: 금액가중수익률(연율 %) 중앙값.
    """
    plan_name: str
    horizon_months: int
    n_windows: int
    loss_prob_pct: float
    median_pct: float
    worst_pct: float
    best_pct: float
    median_mwr_pct: float


class RollingAnalyzer:
    """시작 월을 굴려 가며 납입 계획별 성과 분포를 낸다.

    Args (생성자):
        equity: 일간 자산곡선(시작 1.0).
    """

    def __init__(self, equity: pd.Series):
        self.equity = equity.astype(float)
        # 각 달의 첫 거래일 = 가능한 가입 시점. 투자자는 월 단위로 가입한다.
        self.month_starts = (pd.Series(self.equity.index, index=self.equity.index)
                             .groupby([self.equity.index.year, self.equity.index.month])
                             .min().to_numpy())

    # ── public ──────────────────────────────────────────────────────
    def run(self, plans: Sequence[CashflowPlan], horizons: Sequence[int]) -> List[HorizonStats]:
        """납입 계획 × 보유기간 격자의 롤링 통계를 만든다.

        Args:
            plans: 비교할 납입 계획들.
            horizons: 보유기간(개월) 목록. 구간보다 긴 값은 자동으로 건너뛴다.
        """
        out: List[HorizonStats] = []
        for h in horizons:
            for plan in plans:
                rows = self._windows(plan, h)
                if not rows:
                    logger.warning(f"보유 {h}개월: 표본 창이 없어 건너뜁니다(구간 부족).")
                    continue
                profits = np.array([r[0] for r in rows], dtype=float)
                mwrs = np.array([r[1] for r in rows if r[1] is not None], dtype=float)
                out.append(HorizonStats(
                    plan_name=plan.name,
                    horizon_months=h,
                    n_windows=len(rows),
                    loss_prob_pct=float((profits < 0).mean() * 100.0),
                    median_pct=float(np.median(profits)),
                    worst_pct=float(profits.min()),
                    best_pct=float(profits.max()),
                    median_mwr_pct=float(np.median(mwrs)) if len(mwrs) else float("nan"),
                ))
        return out

    def distribution(self, plan: CashflowPlan, horizon_months: int) -> pd.Series:
        """한 조합의 창별 손익률(%) 원자료(시작일 인덱스). 차트·CSV 용."""
        rows, starts = [], []
        for s in self.month_starts:
            seg = self._segment(s, horizon_months)
            if seg is None:
                continue
            rows.append(DCASimulator(seg).run(plan).profit_pct)
            starts.append(s)
        return pd.Series(rows, index=pd.DatetimeIndex(starts), name=plan.name)

    # ── 내부 ────────────────────────────────────────────────────────
    def _windows(self, plan: CashflowPlan, horizon_months: int):
        """(손익률%, MWR%) 리스트. 구간이 모자란 시작점은 버린다."""
        rows = []
        for s in self.month_starts:
            seg = self._segment(s, horizon_months)
            if seg is None:
                continue
            r = DCASimulator(seg).run(plan)
            rows.append((r.profit_pct, r.mwr_pct))
        return rows

    def _segment(self, start: np.datetime64, horizon_months: int):
        """시작일부터 horizon_months 뒤까지의 곡선 조각. 끝이 구간을 넘으면 None.

        조각은 **시작 1.0 으로 재정규화하지 않는다** — 좌수 계산이 비율만 쓰므로 불필요하다.
        """
        s = pd.Timestamp(start)
        e = s + pd.DateOffset(months=horizon_months)
        if e > self.equity.index[-1]:
            return None                       # 창이 데이터 밖 → 미완성 창은 통계에서 제외
        seg = self.equity.loc[s:e]
        return seg if len(seg) >= 2 else None
