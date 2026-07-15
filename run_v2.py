"""V1(원본) vs V2(변형) 나란히 백테스트 — 결과 비교표 출력.

원본 엔진·설정 파일은 일절 수정하지 않는다. 변형은 세 축으로 준다:
  · indicator — 지표 변형(예: `TrendScoreV2(reversal_weight=…)`)
  · ramp      — 슬롯 경사 진입(예: (50, 60, 0.4) = 50점 40% → 60점 100%)
  · swap      — 유니버스 티커 치환(예: {"0072R0": "411060"} 금 현물 교체)

실행:
    uv run python run_v2.py                      # 전체 기간(config 그대로)
    uv run python run_v2.py --end 2025-12-31     # 기간 컷(config 수정 불필요)

판정 기준(실행 **전에** 고정, 규율):
  Calmar ≥ 기준선 · 최저 해 개선 · 2022 플러스 방어 유지 · CAGR 단독 비교 금지.
  전체 기간과 2025-12-31 컷 **두 구간 모두** 우위여야 채택.
"""
from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from config import Config
from data import ParquetDataLoader
from indicator import Indicator, TrendScoreIndicator
from irp import IRPBacktester, IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_v2")

# 유니버스 치환으로 새로 들어오는 티커의 표시명(원본 irp/config.py 를 건드리지 않기 위해 여기서 보강).
_EXTRA_NAMES: Dict[str, str] = {
    "411060": "ACE KRX금현물",
}


@dataclass(frozen=True)
class Variant:
    """실험 변형 하나.

    Attributes:
        label: 표시명(비교표 행 이름).
        indicator: 순위 산정 지표. 기본은 원본 TrendScore.
        ramp: 슬롯 경사 (경사하단, 만충점수, 하한충전율). None 이면 원본 이진 게이트.
        gate: 신규 편입 자격 문턱. None 이면 경사하단을 따라감(= 문턱까지 같이 내려감).
            60 으로 고정하면 기득권 보호를 유지한 채 크기만 조절한다.
        ramp_hold: 경사를 보유 종목에도 적용할지(False = 부분 '진입'만, 보유는 만충 유지).
        swap: 유니버스 티커 치환 {기존: 신규}. 빈 dict 면 설정 유니버스 그대로.
        drop: 유니버스에서 제외할 티커.
    """
    label: str
    indicator: Indicator = field(default_factory=TrendScoreIndicator)
    ramp: Optional[Tuple[float, float, float]] = None
    gate: Optional[float] = None
    ramp_hold: bool = True
    swap: Mapping[str, str] = field(default_factory=dict)
    drop: Tuple[str, ...] = ()


# ── 실험 조합 (여기만 편집) ─────────────────────────────────────
# Tier 2-a 부분 진입: 60점 이진 진입 → 점수 비례 경사. 전환 구간(5슬롯대, 평균 −1.08%)이
# 유일한 마이너스 상태라는 진단을 정면으로 겨냥한다. 채택하려면 한 점이 아니라 주변값
# 그리드에서 plateau 여야 한다(규율 3) — 그래서 하한충전율·경사폭을 함께 흔든다.
# ── 동결 파라미터 (2026-07-15 확정) ────────────────────────────
# Tier 2-a 부분 진입: 슬롯을 이진(60점=100%)이 아니라 점수 비례로 채운다.
# 52점 30% → 60점 100%. 진입·유지 양쪽에 같은 경사를 적용한다(ramp_hold=True).
#
# 왜 이 점인가 — 하한충전율(0.2~0.4)·문턱(48~52)·만충점수(58~62) 3축 그리드 9점이 **전부**
# 두 구간 모두에서 기준선 Calmar 를 넘겼다(전체 1.04~1.13 / 2025컷 0.88~1.09). 즉 한 점이
# 아니라 면 전체가 들린 plateau 다. 그중 문턱을 8점만 내린 이 점을 고른 이유는 60점 문턱이
# 사실 '기득권 보호' 장치이기 때문 — 많이 내릴수록 조정받은 주도주가 축출된다(2026-04 실측).
FROZEN_RAMP: Tuple[float, float, float] = (52, 60, 0.3)

VARIANTS: List[Variant] = [
    Variant("기준선(이진 60)"),
    Variant("동결 52→60·30%", ramp=FROZEN_RAMP),
]

# plateau 재현용 그리드(규율 3 근거). 필요하면 위 VARIANTS 를 이것으로 바꿔 실행한다.
_PLATEAU_GRID: List[Variant] = [
    Variant("기준선(이진 60)"),
    Variant("50→60 · 20%", ramp=(50, 60, 0.2)),
    Variant("50→60 · 25%", ramp=(50, 60, 0.25)),
    Variant("50→60 · 30%", ramp=(50, 60, 0.3)),
    Variant("50→60 · 35%", ramp=(50, 60, 0.35)),
    Variant("50→60 · 40%", ramp=(50, 60, 0.4)),
    Variant("48→60 · 30%", ramp=(48, 60, 0.3)),
    Variant("52→60 · 30%", ramp=(52, 60, 0.3)),
    Variant("50→58 · 30%", ramp=(50, 58, 0.3)),
    Variant("50→62 · 30%", ramp=(50, 62, 0.3)),
]


# ── 조립 ────────────────────────────────────────────────────────
def _variant_config(icfg: IRPConfig, v: Variant) -> IRPConfig:
    """유니버스를 치환·제외한 IRP 설정 사본을 만든다(원본 설정 객체는 불변 유지)."""
    if not v.swap and not v.drop:
        return icfg
    c = copy.deepcopy(icfg)
    uni = [v.swap.get(t, t) for t in c.satellite.universe if t not in v.drop]
    c.satellite.universe = uni
    for new in v.swap.values():
        if new not in c.satellite.names:
            c.satellite.names[new] = _EXTRA_NAMES.get(new, new)
    return c


def _build(v: Variant, cfg: Config, loader: ParquetDataLoader) -> IRPBacktester:
    """변형 정의에서 백테스터를 만든다. ramp 가 없으면 원본 IRP 백테스터 그대로."""
    if v.ramp is None:
        return IRPBacktester(loader=loader, indicator=v.indicator, cost=cfg.cost)
    lo, full, floor = v.ramp
    sat = SatelliteBacktesterV2(loader=loader, indicator=v.indicator, cost=cfg.cost,
                                ramp_score=lo, full_score=full, entry_gate=v.gate,
                                ramp_floor=floor, ramp_hold=v.ramp_hold)
    return IRPBacktesterV2(loader=loader, indicator=v.indicator, cost=cfg.cost, satellite=sat)


# ── 출력 ────────────────────────────────────────────────────────
def _report(results: Dict[str, object]) -> None:
    """요약 지표표 + 연도별 수익표를 로그로 낸다(정렬 위해 폭 지정 포맷 유지)."""
    logger.info("")
    logger.info(f"{'변형':<20}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'Calmar':>8}{'최저해%':>9}")
    for label, res in results.items():
        m = res.metrics["strategy"]
        worst = min(r["strat_pct"] for r in res.yearly())
        logger.info(f"{label:<20}{m['cagr_pct']:>8.1f}{m['sharpe']:>8.2f}"
                    f"{m['mdd_pct']:>8.1f}{m['calmar']:>8.2f}{worst:>9.1f}")

    logger.info("")
    logger.info("[연도별 수익(%) — 일관성 비교]")
    years = [r["year"] for r in next(iter(results.values())).yearly()]
    logger.info(f"{'변형':<20}" + "".join(f"{y:>9}" for y in years))
    for label, res in results.items():
        by = {r["year"]: r["strat_pct"] for r in res.yearly()}
        logger.info(f"{label:<20}" + "".join(f"{by.get(y, float('nan')):>9.1f}" for y in years))


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP V1/V2 변형 비교 백테스트")
    ap.add_argument("--start", default=None,
                    help="시작일 override(YYYY-MM-DD). 미지정이면 config 값.")
    ap.add_argument("--end", default=None,
                    help="종료일 override(YYYY-MM-DD). config 수정 없이 기간 컷을 하기 위한 인자.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg = Config.load()
    icfg = IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)

    # CLI > IRP 전용 설정 > 파이프라인 공통 설정 순으로 구간을 정한다.
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    logger.info(f"구간 {start} ~ {end} · 변형 {len(VARIANTS)}종")

    results = {}
    for v in VARIANTS:
        bt = _build(v, cfg, loader)
        results[v.label] = bt.run(_variant_config(icfg, v), start=start, end=end)

    _report(results)


if __name__ == "__main__":
    main()
