"""회전율 측정 — 얼마나 자주·크게 갈아탔고, 그 비용이 성과를 얼마나 깎았나.

**왜 필요한가.** 월간 로테이션 + 분기(±7%p) 리밸런싱은 구조적으로 회전을 만든다. 액티브
ETF 제안서에서 "비용은 반영했나"는 반드시 나오는 질문이고, 백테스트가 비용을 넣었다는
말만으로는 부족하다 — 연 몇 %를 돌렸고 그게 수익률 몇 %p 인지 숫자로 나와야 한다.

**어떻게 재나 — 재구현하지 않고 되찾는다(recover).** 엔진은 리밸런싱 시점마다
`total *= (1 - cost × turnover)` 를 곱한다. 그런데 목표비중은 점수·가격만의 함수라
**포트폴리오 금액과 무관**하다. 즉 비용률만 0 으로 바꿔 같은 백테스트를 돌리면 비중 경로가
글자 그대로 같고, 두 자산곡선의 비율이 곱해진 비용 계수만 남는다:

    ratio(t) = eq_cost(t) / eq_free(t) = Π (1 − cost × turnover_k)

따라서 비율의 하루치 계단에서 turnover_k 를 정확히 되찾을 수 있다. 밖에서 비중을 다시
계산해 회전율을 추정하는 방법도 있으나, 그건 엔진과 어긋날 수 있다(`exposure.py` 가 같은
이유로 관측을 택했다). 이 방식은 **엔진이 실제로 청구한 비용** 그 자체를 읽는다.

전제가 깨지면(예: 비용이 비중 경로에 영향을 주게 엔진이 바뀌면) 비율이 오르거나 계단이
1 을 넘는다 — 그때는 조용히 틀린 값을 내는 대신 멈춘다.

**두 계층을 따로 잰다.** 슬리브 안의 월간 로테이션(사테라이트 기준 회전율)과 포트폴리오
상위의 분기 리밸런싱은 기준 금액이 다르다. 슬리브 회전율은 슬리브(=포트폴리오의 70%)
기준이므로, 포트폴리오 환산은 `× satellite_weight` 한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TOL = 1e-9  # 부동소수점 잡음 한계(이보다 작은 회전율은 0으로 본다)


def recover_turnover(equity_cost: pd.Series, equity_free: pd.Series, cost: float,
                     label: str = "") -> pd.Series:
    """비용 있는/없는 두 자산곡선에서 리밸런싱 시점별 단방향 회전율을 되찾는다.

    Args:
        equity_cost: 비용을 반영한 자산곡선.
        equity_free: **같은 설정에 비용률만 0** 으로 둔 자산곡선.
        cost: 왕복 거래비용 비율(예 0.0010).
        label: 오류 메시지에 쓸 대상 표시명.

    Returns:
        회전율이 0 이 아닌 날짜만 담은 Series(값 = 단방향 회전율 0~1).

    Raises:
        ValueError: cost 가 0 이하이거나 두 곡선의 날짜축이 다른 경우.
        RuntimeError: 비용비율이 오르거나(비용이 이득을 준 꼴) 계단이 [0,1] 을 벗어난 경우
            = 비용률 변경이 비중 경로까지 바꿨다는 뜻이므로 이 측정은 성립하지 않는다.
    """
    if cost <= 0:
        raise ValueError("[turnover] cost 가 0 이면 회전율을 되찾을 수 없습니다(계단이 남지 않음).")
    if not equity_cost.index.equals(equity_free.index):
        raise ValueError(f"[turnover] {label} 두 곡선의 날짜축이 다릅니다 — 같은 구간으로 돌리세요.")

    ratio = equity_cost / equity_free
    step = (ratio / ratio.shift(1)).iloc[1:]        # 하루치 비용 계수(비용 없는 날은 1.0)
    turnover = (1.0 - step) / cost

    if float(turnover.min()) < -_TOL / cost:
        raise RuntimeError(
            f"[turnover] {label} 비용비율이 오르는 날이 있습니다(최소 회전율 "
            f"{turnover.min():.3e}) — 비용률 변경이 비중 경로를 바꿨다는 뜻이라 "
            f"이 측정은 성립하지 않습니다.")
    if float(turnover.max()) > 1.0 + 1e-6:
        raise RuntimeError(
            f"[turnover] {label} 단방향 회전율이 1 을 넘습니다({turnover.max():.4f}) — "
            f"대조군 설정이 다릅니다(비용 외 인자가 달라졌는지 확인).")
    return turnover[turnover > _TOL].rename("turnover")


@dataclass(frozen=True)
class TurnoverStats:
    """한 계층(슬리브 로테이션 / 상위 리밸런싱)의 회전율 집계.

    Attributes:
        turnover: 회전이 일어난 날짜별 단방향 회전율(그 계층의 기준 금액 대비).
        cost: 왕복 거래비용 비율.
        scale: 포트폴리오 환산 계수(슬리브면 satellite_weight, 상위면 1.0).
        label: 표시명.
    """
    turnover: pd.Series
    cost: float
    scale: float
    label: str

    @property
    def portfolio_turnover(self) -> pd.Series:
        """포트폴리오 기준으로 환산한 회전율(= 원 회전율 × scale)."""
        return self.turnover * self.scale

    def by_year(self) -> pd.DataFrame:
        """연도별 (회전 횟수 · 단방향 회전율 합% · 비용 드래그%p).

        회전율은 **단방향**이다(업계에서 흔히 쓰는 '연 회전율 100%' = 포트폴리오를 한 번
        갈아엎음과 같은 정의). 비용 드래그는 그 해에 비용으로 빠져나간 금액 비율이며
        `cost × 포트폴리오 회전율` 로 계산한다.
        """
        pt = self.portfolio_turnover
        g = pt.groupby(pt.index.year)
        out = pd.DataFrame({
            "회전횟수": g.size(),
            "회전율%": (g.sum() * 100).round(2),
            "비용드래그%p": (g.sum() * self.cost * 100).round(3),
        })
        out.index.name = "연도"
        return out

    def summary(self, years: float) -> dict:
        """전체 구간 요약 1행(연평균 환산 포함)."""
        total = float(self.portfolio_turnover.sum())
        return {
            "계층": self.label,
            "회전횟수": int(len(self.turnover)),
            "총회전율%": round(total * 100, 1),
            "연평균회전율%": round(total / years * 100, 1),
            "연평균비용드래그%p": round(total / years * self.cost * 100, 3),
        }
