"""국면별 성과 분해 — 상승·하락·횡보장에서 각각 어떻게 벌고 잃었나.

**왜 이 분해가 제안서의 핵심 표인가.** 전체 구간 CAGR·MDD 한 줄로는 "하락 방어형"이라는
상품 주장이 검증되지 않는다. 방어형이라면 **하락장에서** 벤치마크보다 덜 잃어야 하고,
그 대가로 상승장에서는 뒤처져도 된다. 국면을 갈라 봐야 그 교환이 실제로 일어났는지,
아니면 그냥 수익이 낮은 상품인지가 드러난다. 함께 내는 V2 평균 노출이 그 인과를
이어 준다 — 하락장에서 덜 잃었다면 게이트가 실제로 노출을 줄였어야 한다.

**국면 정의(KOSPI200 200일 이동평균).**
  · 상승 — 종가 > 200MA **그리고** 200MA 가 상승 중
  · 하락 — 종가 < 200MA **그리고** 200MA 가 하락 중
  · 횡보 — 나머지(종가와 추세 방향이 엇갈리는 구간 = 전환기)

수준(종가 vs MA) 하나만 쓰면 MA 를 스치는 구간이 상승/하락으로 튀어 국면이 잘게 부서진다.
기울기를 함께 봐서 방향이 확인된 구간만 상승·하락으로 부르고, 애매한 구간은 횡보로 모은다.

**이건 매매 신호가 아니라 사후 라벨이다.** 전략은 이 라벨을 보지 않는다(전략이 쓰는 것은
종목별 TrendScore 뿐). 같은 날 종가로 라벨을 붙이는 것도 그래서 문제가 안 된다 — 성과를
사후에 나눠 읽을 뿐 의사결정에 쓰지 않으므로 룩어헤드가 성립하지 않는다.

**MDD 규약.** 국면 구간만 이어 붙인 합성곡선의 최대낙폭이다. 떨어진 구간 사이의 회복은
빠져 있으므로 '그 국면에 있는 동안 겪은 낙폭'으로 읽어야 하고, 전체 구간 MDD 와 직접
비교하지 않는다(정의가 다르다).
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ANN = 252  # 연율화 기준 거래일 수(BacktestResult 와 동일 규약)

UP, DOWN, FLAT = "상승", "하락", "횡보"
ORDER = [UP, DOWN, FLAT]


def classify_regime(close: pd.Series, index: pd.DatetimeIndex, *,
                    ma_window: int = 200, slope_window: int = 20) -> pd.Series:
    """KOSPI200 종가에서 일별 국면 라벨을 만든다.

    이동평균·기울기는 **자르기 전 전체 시세**로 계산한 뒤 전략 날짜축으로 잘라 낸다.
    먼저 자르면 구간 첫 200봉이 통째로 결측이 돼 2020년이 사라진다(창 처리 실수는 이
    프로젝트에서 이미 사고를 냈다 — `analysis/exposure.py` 주석 참조).

    Args:
        close: 국면 기준 지수(KOSPI200 ETF 069500)의 종가 — 전체 히스토리.
        index: 전략 날짜축(여기에 맞춰 잘라 낸다).
        ma_window: 추세 기준 이동평균 길이(기본 200일).
        slope_window: 이동평균 기울기를 보는 창(기본 20일).

    Raises:
        ValueError: 전략 구간 시작 시점에 이동평균이 아직 정의되지 않은 경우
            (= 기준 지수 히스토리가 워밍업만큼 앞서 있지 않음).
    """
    ma = close.rolling(ma_window).mean()
    slope = ma.diff(slope_window)

    px = close.reindex(index).ffill()
    ma = ma.reindex(index).ffill()
    slope = slope.reindex(index).ffill()
    if ma.isna().any():
        n_bad = int(ma.isna().sum())
        raise ValueError(f"[regime] 구간 앞쪽 {n_bad}일에 {ma_window}일 이동평균이 없습니다 — "
                         f"기준 지수 히스토리가 워밍업만큼 앞서지 않습니다.")

    labels = pd.Series(FLAT, index=index, name="regime")
    labels[(px > ma) & (slope > 0)] = UP
    labels[(px < ma) & (slope < 0)] = DOWN
    return labels


def _curve_stats(equity: pd.Series, mask: pd.Series) -> Dict[str, float]:
    """국면 구간만 이어 붙인 합성곡선의 누적수익·연율화·MDD."""
    ret = equity.pct_change().fillna(0.0)[mask]
    if ret.empty:
        return {"누적수익%": float("nan"), "연율화%": float("nan"), "MDD%": float("nan")}
    curve = (1.0 + ret).cumprod()
    total = float(curve.iloc[-1]) - 1.0
    years = len(ret) / _ANN
    ann = (1.0 + total) ** (1 / years) - 1.0 if years > 0 and total > -1.0 else float("nan")
    mdd = float((curve / curve.cummax() - 1.0).min())
    return {"누적수익%": total * 100, "연율화%": ann * 100, "MDD%": mdd * 100}


def compute_regime_table(curves: Mapping[str, pd.Series], labels: pd.Series,
                         exposure: Optional[pd.Series] = None,
                         exposure_for: Optional[str] = None) -> pd.DataFrame:
    """국면 × 대상 성과표를 만든다(긴 형식 — 국면 한 블록에 대상들이 줄줄이).

    Args:
        curves: {표시명: 자산곡선}. 모두 `labels` 와 같은 날짜축이어야 한다.
        labels: `classify_regime()` 이 낸 일별 국면 라벨.
        exposure: 체크 시점별 포트폴리오 노출(0~1). 국면별 평균을 함께 낸다.
        exposure_for: 노출 열을 채울 대상 표시명(보통 동결 V2). 나머지 행은 빈칸.

    Raises:
        ValueError: 곡선 날짜축이 라벨과 다른 경우.
    """
    for name, curve in curves.items():
        if not curve.index.equals(labels.index):
            raise ValueError(f"[regime] '{name}' 날짜축이 국면 라벨과 다릅니다 — 같은 축으로 맞추세요.")

    total_days = len(labels)
    rows = []
    for regime in ORDER:
        mask = labels == regime
        days = int(mask.sum())
        # 노출은 월 체크 시점에만 관측되므로, 그 체크일의 국면으로 분류해 평균한다.
        exp_mean = float("nan")
        if exposure is not None and days:
            at_check = labels.reindex(exposure.index).ffill()
            sel = exposure[at_check == regime]
            exp_mean = float(sel.mean() * 100) if len(sel) else float("nan")
        for name, curve in curves.items():
            s = _curve_stats(curve, mask)
            rows.append({
                "국면": regime,
                "대상": name,
                "일수": days,
                "비중%": round(days / total_days * 100, 1),
                "누적수익%": round(s["누적수익%"], 2),
                "연율화%": round(s["연율화%"], 2),
                "MDD%": round(s["MDD%"], 2),
                "평균노출%": (round(exp_mean, 1)
                            if (exposure_for and name == exposure_for) else None),
            })
    return pd.DataFrame(rows)


def summary_lines(table: pd.DataFrame) -> list:
    """콘솔용 사람 읽는 요약(국면별 블록, 폭 지정 정렬)."""
    lines = []
    for regime in ORDER:
        blk = table[table["국면"] == regime]
        if blk.empty:
            continue
        head = blk.iloc[0]
        lines.append(f"[{regime}] {int(head['일수'])}일 (전체의 {head['비중%']:.1f}%)")
        lines.append(f"  {'대상':<18}{'누적수익%':>10}{'연율화%':>9}{'MDD%':>8}{'평균노출%':>10}")
        for _, r in blk.iterrows():
            exp = "" if pd.isna(r["평균노출%"]) or r["평균노출%"] is None else f"{r['평균노출%']:.1f}"
            lines.append(f"  {r['대상']:<18}{r['누적수익%']:>10.2f}{r['연율화%']:>9.2f}"
                         f"{r['MDD%']:>8.2f}{exp:>10}")
        lines.append("")
    return lines
