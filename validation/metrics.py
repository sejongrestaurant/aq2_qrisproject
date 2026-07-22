"""자산곡선 조각 → 표준 성과지표. **재구현하지 않는다.**

워크포워드는 하나의 곡선을 창(fold)마다 잘라 각각 평가한다. 그때마다 Calmar·MDD 를 새로
짜면 지표 정의가 두 벌이 되고, 언젠가 한쪽만 고쳐진다(`analysis/segments.py` 가 구간 손익을
엔진 기록에서 그대로 읽는 것과 같은 이유). 그래서 계산은 전부 `BacktestResult._curve_metrics`
에 위임하고, 이 모듈은 **자르고·1.0 으로 되맞추는 일**만 한다.

`_curve_metrics` 는 밑줄로 시작하지만 사실상 이 저장소의 지표 정의 단일 출처다(classmethod 라
결과 객체 없이도 부를 수 있다). 이름을 존중해 접근을 이 파일 한 곳으로 가둔다 — 다른 모듈은
`curve_metrics()` 만 쓴다.

조각 지표를 읽을 때의 주의:
  · CAGR 은 1년짜리 조각에서 사실상 그 해 총수익과 같다(연율화 분모가 1).
  · MDD·Calmar 은 **조각 안에서** 다시 잰다. 창 시작 시점을 새 고점으로 보므로, 창을 걸쳐
    이어지는 낙폭은 두 창에 쪼개져 각각 얕게 보인다. 창별 수치는 창끼리 비교할 때만 쓰고,
    낙폭의 절대 크기는 **이어붙인 전체 OOS 곡선**에서 읽어야 한다.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from backtest import BacktestResult

logger = logging.getLogger(__name__)


def slice_growth(equity: pd.Series, start=None, end=None) -> pd.Series:
    """자산곡선을 [start, end] 로 자르고 시작 1.0 으로 되맞춘 성장곡선을 만든다.

    Args:
        equity: 원 자산곡선(DatetimeIndex).
        start / end: 자를 구간(None 이면 끝까지). 라벨 기준 슬라이스라 경계일을 포함한다.
    Returns:
        시작 1.0 성장곡선.
    Raises:
        ValueError: 구간에 거래일이 2개 미만이라 수익을 낼 수 없는 경우.
    """
    seg = equity.loc[start:end]
    if len(seg) < 2:
        raise ValueError(f"[metrics] 구간 {start}~{end} 의 거래일이 {len(seg)}개뿐이라 "
                         f"성과를 낼 수 없습니다.")
    return (seg / float(seg.iloc[0])).rename("growth")


def curve_metrics(equity: pd.Series) -> Dict[str, float]:
    """성장곡선의 표준 지표(+연도별에서 얻는 '최저 해')를 돌려준다.

    Args:
        equity: 자산곡선. 시작값이 1.0 이 아니어도 알아서 되맞춘다.
    Returns:
        `BacktestResult` 표준 지표(cagr_pct·sharpe·mdd_pct·calmar·sortino·ulcer…)에
        `worst_year_pct`(최저 해 수익률)와 `n_days`(거래일 수)를 더한 dict.
    """
    growth = equity / float(equity.iloc[0])
    m = dict(BacktestResult._curve_metrics(growth, None, None))  # 지표 정의 단일 출처
    years = yearly_returns(growth)
    m["worst_year_pct"] = min(years.values()) if years else float("nan")
    m["n_days"] = float(len(growth))
    return m


# 연도별 집계에서 '해' 로 인정할 최소 거래일 수. 워크포워드 조각은 검증 구간 **직전 거래일**
# (이음매)에서 시작하는데, 그 하루가 이전 연도에 속하면 "거래일 1개짜리 연도, 수익 0.0%" 가
# 만들어진다. 그러면 '최저 해' 가 늘 0.0% 로 고정돼 "잃는 해 없음" 판정이 **무조건 통과**한다
# (실제로 겪은 오탐 — 판정이 아무것도 판정하지 않게 된다). 한 달 미만 조각은 해로 세지 않는다.
_MIN_YEAR_DAYS = 20


def yearly_returns(equity: pd.Series, min_days: int = _MIN_YEAR_DAYS) -> Dict[int, float]:
    """캘린더 연도별 수익률(%)을 돌려준다(부분 연도는 그 구간만의 복리수익).

    일간수익 복리로 계산하므로 첫·마지막 해가 반년만 있어도 정확하다(`BacktestResult.yearly`
    와 같은 규약). 워크포워드에서는 '잃는 해 없음' 주장을 OOS 구간에서 다시 확인하는 데 쓴다.

    Args:
        equity: 자산곡선.
        min_days: 이 거래일 수 미만인 연도는 **버린다**(이음매가 만드는 가짜 연도 방지).
    """
    ret = equity.pct_change().fillna(0.0)
    out: Dict[int, float] = {}
    for y in sorted(set(equity.index.year)):
        seg = ret[ret.index.year == y]
        if len(seg) < min_days:
            continue
        out[int(y)] = float((1.0 + seg).prod() - 1.0) * 100.0
    return out


def chain(segments: list, switch_cost: float = 0.0,
          switch_flags: Optional[list] = None) -> pd.Series:
    """창별 성장곡선 조각을 이어붙여 하나의 연속 곡선으로 만든다.

    각 조각은 시작 1.0 의 성장곡선이다. 이어붙일 때 **첫날은 버린다** — 그날의 수익은 이미
    앞 조각의 마지막 날에 반영돼 있어(조각 경계일이 겹친다) 두 번 세게 된다.

    Args:
        segments: 시작 1.0 성장곡선 리스트(시간순, 구간이 겹치지 않아야 한다).
        switch_cost: 파라미터가 바뀌는 창 경계에서 뺄 비용 비율(왕복 거래비용 × 회전율 가정).
        switch_flags: 조각별 '직전 창과 파라미터가 달라졌는가' bool 리스트(첫 조각은 무시).
            None 이면 비용을 매기지 않는다.
    Returns:
        시작 1.0 의 연속 자산곡선.
    """
    if not segments:
        raise ValueError("[metrics] 이어붙일 조각이 없습니다.")
    parts = [segments[0]]
    level = float(segments[0].iloc[-1])
    for k, seg in enumerate(segments[1:], start=1):
        # 파라미터가 바뀌면 포트폴리오를 새 규칙에 맞춰 재조정해야 한다 → 그 회전율 비용.
        if switch_flags is not None and switch_cost and switch_flags[k]:
            level *= (1.0 - switch_cost)
        parts.append(seg.iloc[1:] * level)      # 경계일 중복 제거
        level *= float(seg.iloc[-1])
    return pd.concat(parts).rename("equity")
