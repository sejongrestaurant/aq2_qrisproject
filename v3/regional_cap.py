"""[C-3] 지역 집중 상한 — 사테라이트 Top-N 선정에 그룹별 슬롯 상한을 건다.

동결 Tier 2-a 슬리브(부분 진입 경사)는 그대로 두고, **어떤 종목을 슬롯에 넣는가**(선정)에만
제약을 추가한다. 점수 내림차순으로 채우되 한 지역 그룹이 이미 상한(cap)만큼 뽑혔으면 그 그룹의
남은 후보는 건너뛴다 → 한 지역(예: 한국섹터 15종)이 Top-7 을 독식하는 것을 막는다.

가설: 무제약이면 특정 국면에 한 지역이 슬롯을 몰아 갖고, 그 지역 동반 하락에 방어가 약해질 수
있다. 그룹 상한은 강제 분산이다. 대가: 상한에 걸려 밀려난 자리에 더 낮은 점수의 타 지역 종목이
들어오므로(품질↓) 추세장 상승을 일부 포기할 수 있다(트레이드오프).

**슬롯 수·충전율 규칙은 불변** — 상한은 오직 '선정 집합' 만 바꾼다. 상한을 넉넉히(그룹당 top_n
이상) 주면 아무 후보도 안 걸러 동결 기준선과 비트 단위로 동일하다(축퇴 보장).

그룹 정의는 `config` 유니버스 구조(글로벌9·미국섹터9·원자재리츠3·한국섹터15 = 36)를 그대로
전사한다. 런타임에 슬리브 유니버스가 전부 어느 그룹에 속하는지 검증한다(미분류 티커가 있으면
조용히 무제약이 되는 대신 즉시 실패시킨다).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from satellite.backtester_v2 import SatelliteBacktesterV2
from satellite.trailing import TrailingStop

logger = logging.getLogger(__name__)

# 세부 4그룹(config 구조 전사 · 라이브 36종). 금은 411060(현물), 죽은 티커(0072R0/0189B0) 제외.
REGION_GROUPS: Dict[str, List[str]] = {
    "글로벌": ["379800", "379810", "453810", "101280", "169950", "283580",
              "099140", "105010", "195930"],
    "미국섹터": ["453650", "200030", "218420", "463640", "463690", "463680",
               "453660", "453630", "453640"],
    "원자재리츠": ["160580", "411060", "329200"],
    "한국섹터": ["117700", "266390", "102960", "091160", "140700", "117460", "140710",
              "091170", "091180", "102970", "117680", "266410", "266420", "266370", "266360"],
}

# 광역 3지역 — 지역·통화 노출 기준. 원자재·금·리츠는 달러표시 해외 분산 자산이라 글로벌축에 귀속.
BROAD_REGION_GROUPS: Dict[str, List[str]] = {
    "미국": REGION_GROUPS["미국섹터"],
    "글로벌": REGION_GROUPS["글로벌"] + REGION_GROUPS["원자재리츠"],
    "한국": REGION_GROUPS["한국섹터"],
}

# 세부 3그룹(실물 제외) — 원자재/금/리츠는 지역이 아니므로 상한에서 면제(uncapped)하고
# 순수 지역 3축(글로벌·미국·한국)에만 상한을 건다. `uncapped=REAL_ASSET_TICKERS` 와 함께 쓴다.
GEO_3GROUPS_EX_REAL: Dict[str, List[str]] = {
    "글로벌": REGION_GROUPS["글로벌"],
    "미국섹터": REGION_GROUPS["미국섹터"],
    "한국섹터": REGION_GROUPS["한국섹터"],
}
REAL_ASSET_TICKERS: List[str] = list(REGION_GROUPS["원자재리츠"])  # 160580·411060·329200


def _build_ticker_group(groups: Dict[str, List[str]]) -> Dict[str, int]:
    """{그룹명: [티커]} → {티커: 그룹인덱스}. 선정 루프에서 O(1) 조회용."""
    return {t: gi for gi, name in enumerate(groups) for t in groups[name]}


# 세부 4그룹 티커→인덱스(기본 그룹핑).
_TICKER_GROUP: Dict[str, int] = _build_ticker_group(REGION_GROUPS)


class RegionalCapBacktester(SatelliteBacktesterV2):
    """그룹별 슬롯 상한을 얹은 동결 Tier 2-a 사테라이트.

    Args (생성자):
        cap: 한 지역 그룹이 가질 수 있는 최대 슬롯 수. top_n 이상이면 무제약(축퇴).
        groups: {그룹명: [티커]} 그룹핑. None 이면 세부 4그룹(`REGION_GROUPS`).
            광역 3지역은 `BROAD_REGION_GROUPS`, 실물 제외 3그룹은 `GEO_3GROUPS_EX_REAL` 를 넘긴다.
        uncapped: 상한에서 **면제**할 티커 목록(그룹 인덱스 −1 = 항상 선정 가능, cap 소비 안 함).
            실물 제외 스킴에서 `REAL_ASSET_TICKERS` 를 넘긴다. 여기 없고 groups 에도 없는 티커는
            '조용한 무제약' 방지를 위해 여전히 즉시 실패시킨다(면제는 명시적이어야 한다).
        나머지: `SatelliteBacktesterV2` 와 동일(동결값 그대로 주입해 선정 제약만 추가).
    """

    def __init__(self, loader, indicator, cost: float = 0.0010, *, cap: int,
                 groups: Optional[Dict[str, List[str]]] = None,
                 uncapped: Optional[List[str]] = None,
                 ramp_score: Optional[float] = None, full_score: Optional[float] = None,
                 entry_gate: Optional[float] = None, ramp_floor: float = 0.4,
                 ramp_hold: bool = True):
        super().__init__(loader=loader, indicator=indicator, cost=cost,
                         ramp_score=ramp_score, full_score=full_score, entry_gate=entry_gate,
                         ramp_floor=ramp_floor, ramp_hold=ramp_hold)
        self.cap = int(cap)
        if self.cap < 1:
            raise ValueError(f"[regional_cap] cap 은 1 이상이어야 합니다: {self.cap}")
        self.groups = groups if groups is not None else REGION_GROUPS
        self._ticker_group: Dict[str, int] = _build_ticker_group(self.groups)
        self._uncapped: set = set(uncapped or [])   # 상한 면제 티커(그룹 −1)
        self._col_group: Optional[np.ndarray] = None  # 컬럼순 그룹 인덱스(_simulate 에서 채움)

    # ── 컬럼→그룹 배열 준비(부모 루프 재사용, 선정만 교체) ─────────────
    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame,
                  atr, cash_ret, top_n, period, trailing: Optional[TrailingStop],
                  entry_score: float, exit_score: float):
        """컬럼 순서에 맞춘 그룹 인덱스 배열을 세팅한 뒤 부모 V2 루프를 그대로 돈다.

        부모 `_simulate` 은 목표비중 산정 시 `self._targets` 를 부르므로, 여기서 컬럼→그룹만
        준비해두면 오버라이드한 `_targets`(상한 선정)가 자동으로 쓰인다. 루프 복제 없음.
        """
        cols = list(closes.columns)
        # 그룹에도 없고 면제 목록에도 없는 티커만 실패시킨다(면제는 −1 로 명시 처리).
        missing = [t for t in cols if t not in self._ticker_group and t not in self._uncapped]
        if missing:  # 미분류 티커 = 조용한 무제약 → 즉시 실패(그룹 정의 갱신 강제)
            raise ValueError(f"[regional_cap] 그룹 미분류 티커: {missing}. 그룹 정의 갱신 필요.")
        self._col_group = np.array([self._ticker_group.get(t, -1) for t in cols], dtype=int)
        return super()._simulate(closes, scores, atr, cash_ret, top_n, period, trailing,
                                 entry_score, exit_score)

    # ── 상한 선정 ────────────────────────────────────────────────────
    def _cap_select(self, ranked: np.ndarray, top_n: int) -> np.ndarray:
        """점수 내림차순 `ranked` 에서 그룹 상한을 지키며 top_n 개를 고른다(그리디).

        그룹 −1(면제 티커)은 상한 판정을 건너뛰고 항상 선정하며 어떤 그룹의 cap 도 소비하지 않는다.
        """
        counts: Dict[int, int] = {}
        out: List[int] = []
        for j in ranked:
            g = int(self._col_group[j])
            if g >= 0:                       # 면제(−1)가 아닌 실제 그룹만 상한 적용
                if counts.get(g, 0) >= self.cap:
                    continue                 # 이 그룹은 이미 상한 도달 → 다음 후보로
                counts[g] = counts.get(g, 0) + 1
            out.append(int(j))
            if len(out) >= top_n:
                break
        return np.array(out, dtype=int)

    def _targets(self, row: np.ndarray, close_row: np.ndarray, held_now: np.ndarray,
                 top_n: int, entry_score: float, exit_score: float
                 ) -> Tuple[np.ndarray, float, np.ndarray]:
        """V2 `_targets` 와 동일하되 선정만 `ranked[:top_n]` → 그룹 상한 그리디로 교체.

        자격 판정(히스테리시스)·충전율 경사·현금 몫은 부모와 완전히 같다. 상한이 top_n 이상이면
        `_cap_select` 가 아무도 안 걸러 부모와 동일 결과가 된다.
        """
        lo, full, gate = self._ramp_bounds(entry_score)
        thr = np.where(held_now, exit_score, gate)
        elig = np.where(~np.isnan(row) & ~np.isnan(close_row) & (row >= thr))[0]
        ranked = elig[np.argsort(row[elig])[::-1]] if len(elig) else elig  # 점수 내림차순
        sel = self._cap_select(ranked, top_n)                              # ★ 그룹 상한

        w_names = np.zeros(len(row))
        if not len(sel):
            return w_names, 1.0, sel
        fills = self._fill(row[sel], lo, full)
        if not self.ramp_hold:
            fills = np.where(held_now[sel], 1.0, fills)
        w_names[sel] = fills / top_n
        w_cash = 1.0 - fills.sum() / top_n
        return w_names, w_cash, sel
