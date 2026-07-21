"""구간 상태별 손익 — 슬롯 수(국면)에 따라 로테이션이 얼마나 벌고 잃나.

동결 V2 사테라이트 슬리브의 로테이션 기록(`rotations_log`)을 입력으로 받아, 각 보유 구간을
'슬롯 수(n)'로 분류하고 구간수익을 집계한다. 두 질문에 답한다:

  ① 구간 손익 전반 — 승률·손익비·Profit Factor(로테이션이 이기는 게임인가).
  ② 슬롯대별 평균 구간수익 — 어떤 국면(슬롯 수)이 돈을 벌고 어떤 국면이 잃나. 특히 진단서가
     지목한 '전환 구간(5슬롯대)'이 부분 진입 도입 후 어떻게 바뀌었나.

**재구현하지 않는다.** 구간 경계·슬롯 수·구간수익은 엔진(`satellite._build_rotations_log`)이
이미 계산한 `rotations_log`(각 원소 `{"date","labels","n","ret_pct"}`)를 그대로 읽는다.
`n` 은 소비된 슬롯 수(충전율 무관 — 30%만 채운 슬롯도 1로 센다), `ret_pct` 는 그 교체일부터
다음 교체까지 슬리브 자산곡선의 변화다.

**소표본 주의.** 슬롯대별 구간 수는 크게 치우친다(만충 n=7 이 대부분, 전환대 n=4~6 은 각
몇 개뿐). 슬롯대별 평균은 표본 수를 반드시 함께 봐야 한다 — 이 모듈은 늘 `n_segments` 를
같이 낸다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentStats:
    """구간 상태별 손익 집계.

    Attributes:
        by_slot: 슬롯 수(n)별 (구간 수, 평균 구간수익%) 표.
        n_segments: 전체 구간 수.
        win_pct: 승률(구간수익 > 0 비율, %).
        payoff: 손익비 = 평균이익 / |평균손실|.
        profit_factor: Σ이익 / |Σ손실|.
        avg_pct / best_pct / worst_pct: 구간수익 평균·최고·최저(%).
    """
    by_slot: pd.DataFrame
    n_segments: int
    win_pct: float
    payoff: float
    profit_factor: float
    avg_pct: float
    best_pct: float
    worst_pct: float

    def summary_lines(self, top_n: int) -> List[str]:
        """로그·콘솔용 사람 읽는 요약(정렬 위해 폭 지정)."""
        lines = [
            f"구간 손익(전체 {self.n_segments}구간): "
            f"승률 {self.win_pct:.0f}% · 손익비 {self.payoff:.2f} · PF {self.profit_factor:.2f}",
            f"  평균 {self.avg_pct:+.2f}% · 최고 {self.best_pct:+.2f}% · 최저 {self.worst_pct:+.2f}%",
            "슬롯대(n)별 평균 구간수익 — 표본 수(구간)를 함께 볼 것:",
        ]
        for n, row in self.by_slot.iterrows():
            state = "만충" if n == top_n else ("현금방어" if n == 0 else "슬롯대")
            lines.append(f"  n={int(n)} {state:<5} 구간 {int(row['n_segments']):>2}개 · "
                         f"평균 {row['mean_pct']:+6.2f}%")
        return lines


def compute_segment_stats(rotations_log: List[dict]) -> SegmentStats:
    """`rotations_log` 에서 구간 상태별 손익을 집계한다.

    Args:
        rotations_log: 엔진이 낸 교체 기록(각 원소 `n`·`ret_pct` 포함).
    Raises:
        ValueError: 기록이 비어 있어 집계할 구간이 없는 경우.
    """
    if not rotations_log:
        raise ValueError("[segments] rotations_log 가 비어 집계할 구간이 없습니다.")
    df = pd.DataFrame([{"n": r["n"], "ret": r["ret_pct"] / 100.0} for r in rotations_log])

    by_slot = df.groupby("n")["ret"].agg(n_segments="size", mean="mean")
    by_slot["mean_pct"] = (by_slot["mean"] * 100).round(2)
    by_slot = by_slot[["n_segments", "mean_pct"]]

    r = df["ret"].to_numpy()
    pos, neg = r[r > 0], r[r < 0]
    # 손익비·PF 의 경계 처리. 두 극단을 구분한다:
    #   · 손실 구간 0 → 분모가 0(∞). 소표본에서 실제로 생길 수 있어 NaN 으로 둔다('무한대'를
    #     수치로 찍어 오해를 부르지 않게).
    #   · 이익 구간 0(전패) → 분자가 0 이므로 손익비·PF 는 0(명백히 지는 게임). NaN 이 아니다.
    if not len(neg):
        payoff = pf = float("nan")
    elif not len(pos):
        payoff = pf = 0.0
    else:
        payoff = float(pos.mean() / abs(neg.mean()))
        pf = float(pos.sum() / abs(neg.sum()))
    return SegmentStats(
        by_slot=by_slot,
        n_segments=int(len(r)),
        win_pct=float((r > 0).mean() * 100),
        payoff=payoff,
        profit_factor=pf,
        avg_pct=float(r.mean() * 100),
        best_pct=float(r.max() * 100),
        worst_pct=float(r.min() * 100),
    )
