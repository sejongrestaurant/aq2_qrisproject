"""후보 파라미터 격자 + 후보별 전 구간 곡선 산출.

## 무엇을 '학습' 하는가

이 전략에는 적합(fit)할 모형이 없다. 대신 **고르는 것**이 있다 — Tier 2-a 슬롯 경사의 세 축
(경사 하단 문턱 · 만충 점수 · 하한 충전율)이다. 동결값 52/60/0.3 은 이 격자를 전 구간에서
보고 고른 값이므로, 워크포워드가 검증할 대상도 정확히 **이 선정 단계**다.

**검증하지 않는 것(정직하게 명시):** top_n=7, 히스테리시스 60/45, TrendScore 가중치
(EWMAC 0.55 · TSMOM 0.25 · RSI 0.20), 채권 30% 고정, 유니버스 36종은 여기서 재선정하지
않는다. 이들도 전 구간을 보고 정한 값이므로, 이 워크포워드는 '설계 전체가 OOS 에서
살아남는가' 가 아니라 **'동결 선정이 표본 밖에서 재현되는가'** 를 답한다. 전자를 물으려면
전략 설계 자체를 학습 창 안에서 다시 세워야 하는데 그건 다른 프로젝트다.

## 전 구간 한 번 돌리고 창마다 자른다 — 왜 정당한가

엔진은 **인과적**이다: t 시점 상태는 t 이하의 시세와 파라미터만으로 결정된다(체결은 다음
거래일 반영, 지표는 워밍업 NaN). 그러므로 후보 c 를 2020~2026 전 구간으로 한 번 돌린 곡선을
[t0, t1] 로 자른 것은, 같은 후보를 t1 까지만 돌려 얻은 곡선의 그 구간과 **완전히 같다**.
창마다 다시 돌릴 필요가 없다 — 후보 수만큼만 돌리면 된다(46개 × 2초).

**남는 한계 하나(측정하지 않았다):** 실제 운용이라면 창이 바뀔 때 직전 파라미터로 굴러온
포지션을 새 파라미터에 맞춰 재조정한다. 여기서는 각 후보가 '처음부터 그 파라미터로 굴러온'
상태에서 창을 자르므로, 창 경계의 보유 상태가 실제와 다르다. 월간 로테이션이라 한 달이면
씻겨 나가고, 경계 회전율은 `switch_cost` 로 대략 물리지만(`metrics.chain`), 정확한 값은
아니다. 엔진에 상태 인계 자리가 없어 그대로 두고 한계로 남긴다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data import DataLoader
from indicator import Indicator
from irp.backtester_v2 import IRPBacktesterV2
from irp.config import IRPConfig
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger(__name__)

# 동결 근거로 쓰인 3축 그리드의 축 값(CLAUDE.md '완료된 실험' 의 하한충전율·문턱·만충점수).
# 워크포워드는 이 격자를 창마다 다시 훑는다 — 같은 격자를 써야 '그때 이 규칙이었다면' 이 된다.
AXES_FULL: Dict[str, Tuple[float, ...]] = {
    "ramp_score": (48.0, 50.0, 52.0),
    "full_score": (58.0, 60.0, 62.0),
    "ramp_floor": (0.20, 0.25, 0.30, 0.35, 0.40),
}
# 빠른 확인용 축소 격자(구조 점검·디버깅). 결론용으로 쓰지 말 것.
AXES_QUICK: Dict[str, Tuple[float, ...]] = {
    "ramp_score": (50.0, 52.0),
    "full_score": (60.0,),
    "ramp_floor": (0.25, 0.30, 0.35),
}

# V1 원설계(이진 게이트 60) 라벨. 격자에 **함께 넣는다** — 워크포워드가 어떤 창에서
# 'Tier 2-a 를 쓰지 않는 편이 낫다' 고 판단할 자유를 주기 위해서다(귀무가설을 후보로).
BINARY_LABEL = "V1 이진60"


@dataclass(frozen=True)
class Candidate:
    """후보 파라미터 하나.

    Attributes:
        label: 표시명(모든 표·차트의 키).
        ramp: (경사 하단, 만충 점수, 하한 충전율). None 이면 V1 이진 게이트.
        coord: 격자 좌표(축별 인덱스). 이웃을 찾는 데 쓴다. 격자 밖 후보(V1)는 None.
    """
    label: str
    ramp: Optional[Tuple[float, float, float]]
    coord: Optional[Tuple[int, int, int]] = None

    @property
    def is_binary(self) -> bool:
        """V1 이진 게이트 후보인지."""
        return self.ramp is None


class CandidateGrid:
    """후보 격자 — 조합 생성과 '이웃' 정의를 한곳에서 맡는다.

    이웃 개념이 여기 있는 이유: 규율 3(주변값 강건성)은 "한 점만 좋으면 기각" 이라고 말한다.
    그 규율을 선정 규칙으로 코드화하려면(`selection.PlateauRule`) 무엇이 주변값인지 정의가
    필요한데, 그건 격자의 구조를 아는 이 클래스의 책임이다.

    Args (생성자):
        axes: 축 이름 → 값 튜플. 기본은 동결 근거 그리드와 같은 3축.
        include_binary: V1 이진 게이트를 후보에 포함할지.
    """

    def __init__(self, axes: Optional[Dict[str, Sequence[float]]] = None,
                 include_binary: bool = True):
        self.axes = {k: tuple(v) for k, v in (axes or AXES_FULL).items()}
        self.include_binary = bool(include_binary)
        self.candidates: List[Candidate] = self._build()
        self._by_label = {c.label: c for c in self.candidates}
        logger.info(f"후보 격자 {len(self.candidates)}개 · "
                    f"경사하단 {self.axes['ramp_score']} · 만충 {self.axes['full_score']} · "
                    f"하한충전 {self.axes['ramp_floor']}"
                    + (f" + {BINARY_LABEL}" if self.include_binary else ""))

    # ── public ──────────────────────────────────────────────────────
    def get(self, label: str) -> Candidate:
        """라벨로 후보를 찾는다."""
        if label not in self._by_label:
            raise KeyError(f"[grid] 후보 '{label}' 이 격자에 없습니다.")
        return self._by_label[label]

    def neighbors(self, cand: Candidate) -> List[Candidate]:
        """격자에서 **한 축만 한 칸** 다른 후보들(자기 자신 제외).

        V1 처럼 격자 밖 후보는 이웃이 없다 — 빈 리스트를 준다. 그러면 plateau 규칙에서
        자기 점수만으로 평가되어, '주변이 평평한지' 를 물을 수 없는 후보가 특별히
        유리해지지도 불리해지지도 않는다.
        """
        if cand.coord is None:
            return []
        out: List[Candidate] = []
        keys = list(self.axes.keys())
        for ax, ci in enumerate(cand.coord):
            for delta in (-1, 1):
                nj = ci + delta
                if 0 <= nj < len(self.axes[keys[ax]]):
                    coord = list(cand.coord)
                    coord[ax] = nj
                    hit = self._at(tuple(coord))
                    if hit is not None:
                        out.append(hit)
        return out

    def label_of(self, ramp: Optional[Tuple[float, float, float]]) -> str:
        """파라미터 튜플에 대응하는 격자 후보의 라벨(없으면 KeyError)."""
        for c in self.candidates:
            if c.ramp == ramp:
                return c.label
        raise KeyError(f"[grid] 파라미터 {ramp} 가 격자에 없습니다 — 축 값을 확인하세요.")

    # ── 내부 ────────────────────────────────────────────────────────
    def _build(self) -> List[Candidate]:
        out: List[Candidate] = []
        los, fulls, floors = (self.axes["ramp_score"], self.axes["full_score"],
                              self.axes["ramp_floor"])
        for i, lo in enumerate(los):
            for j, full in enumerate(fulls):
                if full < lo:
                    continue                    # 만충 < 경사하단 = 정의되지 않는 조합
                for k, floor in enumerate(floors):
                    out.append(Candidate(label=f"{lo:.0f}→{full:.0f}·{floor * 100:.0f}%",
                                         ramp=(lo, full, floor), coord=(i, j, k)))
        if self.include_binary:
            out.append(Candidate(label=BINARY_LABEL, ramp=None, coord=None))
        return out

    def _at(self, coord: Tuple[int, int, int]) -> Optional[Candidate]:
        for c in self.candidates:
            if c.coord == coord:
                return c
        return None


class CandidateRunner:
    """후보별로 IRP 백테스트를 한 번씩 돌려 일간 자산곡선을 캐시한다.

    Args (생성자):
        loader: 시세 로더(메모이제이션 래퍼를 넣는 것을 권장 — `memo.MemoLoader`).
        indicator: 순위 산정 지표(마찬가지로 `memo.MemoIndicator` 권장).
        cost: 왕복 거래비용 비율.
        allow_missing: 유니버스 종목이 빠져도 진행할지(기본 False = fail-loud).

    Attributes:
        curves: 라벨 → 일간 자산곡선(시작 1.0).
        benchmark: 벤치마크(TRF7030) 곡선. 후보와 무관하므로 첫 실행에서 한 번만 챙긴다.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010,
                 allow_missing: bool = False):
        self.loader = loader
        self.indicator = indicator
        self.cost = cost
        self.allow_missing = bool(allow_missing)
        self.curves: Dict[str, pd.Series] = {}
        self.benchmark: Optional[pd.Series] = None
        self.benchmark_name: str = "벤치마크"

    def run_all(self, icfg: IRPConfig, candidates: Sequence[Candidate],
                start=None, end=None) -> Dict[str, pd.Series]:
        """후보 전부를 전 구간으로 돌려 곡선을 모은다.

        실패는 **삼키지 않는다**. 후보 하나가 조용히 빠지면 그 창의 선정 대상이 달라져
        결과가 설명 불가능해진다(`data/integrity.py` 가 종목에 대해 하는 것과 같은 원칙).

        Returns:
            라벨 → 일간 자산곡선. 모든 곡선은 같은 거래일 인덱스를 갖는다(같은 구간·같은
            유니버스라 정렬이 일치) — 다르면 창 자르기가 어긋나므로 확인 후 예외를 던진다.
        """
        for n, cand in enumerate(candidates, start=1):
            res = self._build(cand).run(icfg, start=start, end=end)
            self.curves[cand.label] = res.equity
            if self.benchmark is None:
                self.benchmark = res.benchmark
                self.benchmark_name = res.benchmark_name
            logger.info(f"  [{n:>2}/{len(candidates)}] {cand.label:<14} "
                        f"CAGR {res.metrics['strategy']['cagr_pct']:>5.1f}% · "
                        f"Calmar {res.metrics['strategy']['calmar']:.2f}")
        self._check_aligned()
        return self.curves

    # ── 내부 ────────────────────────────────────────────────────────
    def _build(self, cand: Candidate) -> IRPBacktesterV2:
        """후보 정의에서 백테스터를 만든다(`run_v2._build` 와 같은 조립 규약).

        V1 후보도 V2 백테스터로 감싼다 — 슬리브를 주지 않으면 원본 `SatelliteBacktester` 를
        그대로 쓰므로 동작은 V1 이고, 유니버스 무결성 가드만 함께 걸린다.
        """
        common = dict(loader=self.loader, indicator=self.indicator, cost=self.cost,
                      allow_missing=self.allow_missing)
        if cand.is_binary:
            return IRPBacktesterV2(**common)
        lo, full, floor = cand.ramp
        sat = SatelliteBacktesterV2(loader=self.loader, indicator=self.indicator, cost=self.cost,
                                    ramp_score=lo, full_score=full, ramp_floor=floor,
                                    ramp_hold=True)   # 동결 구성과 동일(진입·유지 같은 경사)
        return IRPBacktesterV2(satellite=sat, **common)

    def _check_aligned(self) -> None:
        """모든 후보 곡선의 거래일 인덱스가 같은지 확인한다(다르면 창 자르기가 어긋난다)."""
        ref_label, ref = next(iter(self.curves.items()))
        for label, curve in self.curves.items():
            if not curve.index.equals(ref.index):
                raise ValueError(
                    f"[runner] 후보 '{label}' 의 거래일 인덱스가 '{ref_label}' 과 다릅니다 "
                    f"({len(curve)} vs {len(ref)}봉). 같은 구간·같은 유니버스에서 나온 곡선이 "
                    f"아니면 창별 비교가 성립하지 않습니다.")
