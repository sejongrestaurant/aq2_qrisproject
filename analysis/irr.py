"""불규칙 현금흐름의 내부수익률(XIRR).

적립식은 돈이 **여러 시점에 나눠** 들어간다. 그래서 자산곡선의 CAGR(시간가중수익률, TWR)은
투자자가 실제로 번 돈을 말해 주지 못한다 — TWR 은 '1원을 처음에 넣고 끝까지 뒀을 때'의
수익률이고, 납입 타이밍을 무시한다. 투자자 체감 수익률은 **금액가중수익률(MWR = XIRR)** 이다.

예: 곡선이 초반에 폭등하고 후반에 횡보하면 TWR 은 높지만, 적립식 투자자는 폭등 구간에 넣은
돈이 적어 MWR 이 훨씬 낮다. 제안서에서 두 수치를 **나란히** 제시해야 정직하다.

외부 의존(scipy·numpy_financial) 없이 이분법으로 푼다. 현금흐름 부호 규약은 통상의 IRR 과
같다: 납입(유출)은 음수, 최종 평가액(유입)은 양수.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.0
# 이분법 탐색 범위: 연 -99.99% ~ +1000%. 실무상 이 밖으로 나가는 해는 무의미하다.
_LO, _HI = -0.9999, 10.0
_TOL, _MAX_ITER = 1e-9, 200


def _npv(rate: float, years: np.ndarray, amounts: np.ndarray) -> float:
    """연율 rate 로 할인한 현재가치 합."""
    return float(np.sum(amounts / np.power(1.0 + rate, years)))


def xirr(cashflows: pd.Series) -> Optional[float]:
    """불규칙 현금흐름의 연율 내부수익률을 구한다.

    Args:
        cashflows: 날짜 인덱스 Series. 납입은 음수, 회수(최종 평가액)는 양수.
    Returns:
        연율 수익률(0.07 = 7%). 해가 탐색 범위 밖이거나 부호가 한쪽뿐이면 None.
    """
    if len(cashflows) < 2:
        return None
    amounts = cashflows.to_numpy(dtype=float)
    if not (amounts.max() > 0 and amounts.min() < 0):
        return None  # 유입·유출이 모두 있어야 IRR 이 정의된다

    t0 = cashflows.index[0]
    years = np.array([(d - t0).days / _DAYS_PER_YEAR for d in cashflows.index], dtype=float)

    lo, hi = _LO, _HI
    f_lo, f_hi = _npv(lo, years, amounts), _npv(hi, years, amounts)
    if f_lo * f_hi > 0:
        logger.warning("XIRR: 탐색 범위에서 부호 변화가 없어 해를 찾지 못했습니다.")
        return None

    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2.0
        f_mid = _npv(mid, years, amounts)
        if abs(f_mid) < _TOL or (hi - lo) < _TOL:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0
