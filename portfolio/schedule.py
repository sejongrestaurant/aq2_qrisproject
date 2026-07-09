"""리밸런싱·체크 주기 스케줄 헬퍼(포트폴리오·사테라이트 공용).

주기 문자열을 거래일 마스크로 바꾸는 `period_mask` 와, 자산곡선을 경계일로 잘라 구간별
보유거래(`Trade`)로 만드는 `segment_trades` 를 제공한다. 두 백테스터가 동일 규약을 공유하도록
한곳에 모았다(중복 방지).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from backtest import Trade

logger = logging.getLogger(__name__)


def period_mask(dates: pd.DatetimeIndex, period: str | None) -> np.ndarray:
    """주기 경계가 되는 거래일을 True 로 표시한다.

    period: "D"(매 거래일)·"W"(주)·"M"(월)·"Q"(분기)·"Y"(연)·"<N>D"(N거래일).
    달력 단위는 그 단위가 바뀌는 첫 거래일을 True 로 둔다. None/미지원 값이면 전부 False.
    """
    n = len(dates)
    mask = np.zeros(n, dtype=bool)
    if not period:
        return mask
    p = str(period).strip().upper()
    if p in ("NONE", "OFF"):
        return mask
    if p in ("D", "DAY", "DAILY", "1D"):
        mask[1:] = True  # 첫날(진입 전) 제외 매 거래일 체크
        return mask
    if p.endswith("D") and p[:-1].isdigit():
        step = int(p[:-1])
        if step > 0:
            mask[step::step] = True
        return mask

    # 달력 경계(월/분기/연/주)로 그룹키를 만들고, 키가 바뀌는 첫 거래일을 표시
    key_fn = {
        "M": lambda d: (d.year, d.month),
        "Q": lambda d: (d.year, (d.month - 1) // 3),
        "Y": lambda d: (d.year,),
        "A": lambda d: (d.year,),
        "W": lambda d: d.isocalendar()[:2],  # (ISO 연도, ISO 주차)
    }.get(p)
    if key_fn is None:
        logger.warning(f"알 수 없는 주기 '{period}' → 주기 트리거 없이 진행")
        return mask

    prev = None
    for i, d in enumerate(dates):
        k = key_fn(d)
        if prev is not None and k != prev:
            mask[i] = True
        prev = k
    return mask


def segment_trades(equity: pd.Series, boundary_dates: List[pd.Timestamp],
                   reason: str = "리밸런싱") -> List[Trade]:
    """자산곡선을 경계일로 잘라 각 보유구간을 하나의 `Trade` 로 만든다.

    리밸런싱/종목교체처럼 개별 왕복거래가 없는 백테스트에서, 활동(구간 수·구간수익률·승률)을
    리포트 거래 테이블로 보여주기 위한 표현이다. 마지막 구간의 청산 사유는 'eod'.

    Args:
        equity: 시작 1.0 자산곡선(DatetimeIndex).
        boundary_dates: 구간을 나누는 경계일(리밸런싱·교체가 일어난 날).
        reason: 중간 경계의 청산 사유 라벨(예: "리밸런싱"·"교체").
    """
    idx = equity.index
    bounds = [idx[0]] + [d for d in boundary_dates if d not in (idx[0], idx[-1])] + [idx[-1]]
    trades: List[Trade] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        pa, pb = idx.get_loc(a), idx.get_loc(b)
        eq_a, eq_b = float(equity.iloc[pa]), float(equity.iloc[pb])
        trades.append(Trade(
            entry_date=a, exit_date=b, entry_px=eq_a, exit_px=eq_b,
            ret=eq_b / eq_a - 1.0, bars_held=pb - pa,
            exit_reason=("eod" if b == idx[-1] else reason)))
    return trades
