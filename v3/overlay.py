"""[B-1] 변동성 타깃 오버레이 — 포트 실현변동성이 목표를 넘으면 위험노출을 비례 축소.

이미 완성된 IRP 자산곡선 위에 얹는 **사후 오버레이**다(엔진 무수정). 매일 직전까지의 60거래일
실현변동성(연율)을 재고, 목표를 초과하면 그 초과분만큼 다음날 노출을 줄인다:

    레버리지 L_t = min(1, 목표변동성 / 실현변동성_{t-1})   (≤1 — 축소만, 확대 없음)
    조정수익_t = L_t · 전략수익_t + (1 − L_t) · 현금수익_t

'축소만' 인 이유: 목표 이하일 때 레버리지를 1 초과로 키우면 상승장에 위험을 더 지는 셈이라
하락 방어형 상품 철학과 어긋난다. 못 실은 (1−L) 몫은 현금 대용(단기채 153130) 수익을 받는다.

룩어헤드 방지: t 일 노출은 t−1 일까지의 변동성으로 정한다(당일 변동성으로 당일을 줄이지 않음).
워밍업(첫 window 일)은 변동성 추정이 안 되므로 L=1(원곡선 그대로).

예상 실패 모드(전달문): 게이트와 **이중 감속**. 게이트가 이미 약세장에서 현금비중을 높이는데
변동성 오버레이가 또 줄이면, 반등 초입 상승을 두 번 깎아 CAGR·Calmar 가 함께 내려갈 수 있다.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import BacktestResult

logger = logging.getLogger(__name__)

_ANN = 252  # 연율화 거래일


def vol_target(res: BacktestResult, cash_ret: pd.Series,
               target_ann: float, window: int = 60) -> BacktestResult:
    """전략 자산곡선에 변동성 타깃 오버레이를 적용한 새 `BacktestResult` 를 만든다.

    Args:
        res: 원 IRP 백테스트 결과(동결 V2).
        cash_ret: 현금 대용(단기채 153130) 일간수익, res.equity 날짜축에 정렬 가능해야 함.
        target_ann: 목표 연변동성(예 0.10 = 10%).
        window: 실현변동성 추정 창(거래일). 기본 60.

    Returns:
        equity 만 오버레이로 갈아끼운 결과(benchmark·trades·price 등은 원본 재사용).
    """
    eq = res.equity
    ret = eq.pct_change().fillna(0.0)
    cash = cash_ret.reindex(eq.index).fillna(0.0)

    # 직전까지의 실현변동성(연율). shift(1) 로 당일 정보를 배제(룩어헤드 방지).
    realized = ret.rolling(window).std().shift(1) * np.sqrt(_ANN)
    lev = (target_ann / realized).clip(upper=1.0)   # 축소만(≤1)
    lev = lev.fillna(1.0)                            # 워밍업 구간은 원노출

    scaled = lev * ret + (1.0 - lev) * cash
    new_eq = (1.0 + scaled).cumprod().rename("equity")
    new_eq.iloc[0] = 1.0

    avg_lev = float(lev.mean())
    logger.info(f"변동성타깃 {target_ann * 100:.0f}% · 창 {window}일 · 평균 레버리지 "
                f"{avg_lev:.2f}(=평균 실효노출) · 감속일 {int((lev < 0.999).sum())}/{len(lev)}")
    return replace(res, equity=new_eq, _metrics=None)
