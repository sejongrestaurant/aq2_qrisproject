"""적립식 시뮬레이터 — 자산곡선 + 납입 계획 → 실제 투자자 손익.

전략 엔진은 '1원을 넣고 굴렸을 때' 의 곡선(equity, 시작 1.0)만 만든다. 투자자는 거기에
매달 돈을 더 넣는다. 이 모듈은 그 둘을 합쳐 **총 납입액 대비 최종 평가액**을 낸다.

계산 규약 — 자산곡선을 '기준가(NAV)' 로 본다:
    납입일 t 에 amount 를 넣으면 `amount / equity[t]` 좌수를 산다.
    최종 평가액 = (총 좌수) × equity[마지막날].
이렇게 하면 곡선의 수익률 구조를 그대로 쓰면서 납입 타이밍만 얹을 수 있다(엔진 무수정).
매수 수수료는 곡선에 이미 반영된 전략 비용과 성격이 달라 여기서 다루지 않는다 — IRP 계좌의
ETF 매수 수수료는 증권사별로 0에 가깝고, 모델에 넣으면 근거 없는 정밀도를 준다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .cashflow import CashflowPlan
from .irr import xirr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DCAResult:
    """적립식 시뮬레이션 결과.

    Attributes:
        plan_name: 납입 계획 표시명.
        start / end: 투자 구간(거래일).
        n_payments: 납입 횟수.
        contributed: 총 납입액(원).
        final_value: 최종 평가액(원).
        profit: 손익(원) = final_value − contributed.
        profit_pct: 총 납입액 대비 손익률(%). 납입 시점이 제각각이라 '연 수익률'이 아니다.
        mwr_pct: 금액가중수익률(XIRR, 연율 %). 투자자 체감 수익률. 해가 없으면 None.
        twr_pct: 시간가중수익률(곡선 CAGR, 연율 %). 전략 자체의 성적(납입 타이밍 무관).
    """
    plan_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_payments: int
    contributed: float
    final_value: float
    profit: float
    profit_pct: float
    mwr_pct: Optional[float]
    twr_pct: float


class DCASimulator:
    """자산곡선 위에 납입 현금흐름을 얹어 투자자 손익을 계산한다.

    Args (생성자):
        equity: 일간 자산곡선(시작 1.0). `BacktestResult.equity` 를 그대로 넣는다.
    """

    def __init__(self, equity: pd.Series):
        if len(equity) < 2:
            raise ValueError("적립식 시뮬레이션: 자산곡선이 너무 짧습니다(2일 미만).")
        self.equity = equity.astype(float)

    # ── public ──────────────────────────────────────────────────────
    def run(self, plan: CashflowPlan) -> DCAResult:
        """납입 계획 하나를 시뮬레이션한다."""
        idx = self.equity.index
        contrib = plan.generate(idx)
        if contrib.empty:
            raise ValueError(f"{plan.name}: 구간 내 납입일이 없습니다.")

        nav = self.equity.reindex(contrib.index)
        if nav.isna().any():
            raise ValueError(f"{plan.name}: 납입일이 자산곡선 밖입니다(정렬 오류).")

        units = (contrib / nav).sum()
        final = float(units * self.equity.iloc[-1])
        total = float(contrib.sum())
        profit = final - total

        return DCAResult(
            plan_name=plan.name,
            start=idx[0], end=idx[-1],
            n_payments=len(contrib),
            contributed=total,
            final_value=final,
            profit=profit,
            profit_pct=profit / total * 100.0,
            mwr_pct=self._mwr(contrib, final, idx[-1]),
            twr_pct=self._twr(),
        )

    def curve(self, plan: CashflowPlan) -> pd.DataFrame:
        """일별 '납입 누계(원금) vs 평가액' 추이. 제안서 표지급 그림의 원자료.

        Returns:
            DataFrame(index=거래일, columns=[paid, value]). paid 는 그날까지 넣은 원금 누계,
            value 는 그날까지 산 좌수의 그날 평가액. 둘의 간격이 곧 수익이다.
        """
        idx = self.equity.index
        contrib = plan.generate(idx)
        paid = contrib.reindex(idx).fillna(0.0).cumsum()
        bought = (contrib / self.equity.reindex(contrib.index))  # 납입일마다 산 좌수
        units = bought.reindex(idx).fillna(0.0).cumsum()
        return pd.DataFrame({"paid": paid, "value": units * self.equity})

    # ── 내부 ────────────────────────────────────────────────────────
    @staticmethod
    def _mwr(contrib: pd.Series, final: float, end: pd.Timestamp) -> Optional[float]:
        """금액가중수익률(XIRR). 납입은 유출(−), 최종 평가액은 마지막날 유입(+)."""
        flows = (-contrib).copy()
        # 마지막 납입일과 종료일이 같으면 한 날짜에 두 흐름 → 합산해야 인덱스가 안 겹친다.
        flows.loc[end] = flows.get(end, 0.0) + final
        r = xirr(flows.sort_index())
        return None if r is None else r * 100.0

    def _twr(self) -> float:
        """시간가중수익률(곡선 CAGR, 연율 %). 납입 타이밍과 무관한 전략 자체 성적."""
        years = (self.equity.index[-1] - self.equity.index[0]).days / 365.0
        if years <= 0:
            return 0.0
        return ((self.equity.iloc[-1] / self.equity.iloc[0]) ** (1.0 / years) - 1.0) * 100.0
