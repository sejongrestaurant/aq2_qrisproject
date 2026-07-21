"""실효 노출률 러너 — 동결 V2 가 실제로 얼마나 투자돼 있었나(제안서 전시물).

구 지표 '슬롯 미달 개월 수'를 대체한다. 그 지표는 V1(이진 게이트)의 언어라 부분 충전이 있는
동결 V2 에서는 정의가 어긋나고 노출을 과대평가한다(30%만 채운 슬롯도 '찼다'로 센다).

실행:
    uv run python run_exposure.py                     # 전체 기간
    uv run python run_exposure.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    exposure_monthly.csv   체크 시점별 노출·소비슬롯·유효후보·자격통과·상태
    exposure_monthly.png   V1 계단 vs V2 경사로

**측정 신뢰성 — 프로브가 결과를 흔들지 않음을 매 실행 검증한다.** 관측용 프로브를 끼운
백테스트와 끼우지 않은 대조군의 자산곡선을 대조해, 다르면 멈춘다. 이게 통과해야 "이 노출은
동결 상품이 실제로 쓴 값"이라고 말할 수 있다. 덤으로 V1 축퇴 보장(`ramp_score=None` 이면
원본과 부동소수점까지 동일)도 매번 재확인된다.
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import pandas as pd

from analysis.exposure import ExposureProbe, ExposureResult
from analysis.exposure_report import ExposureReport
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_exposure")

Ramp = Optional[Tuple[float, float, float]]


# ── 조립 ────────────────────────────────────────────────────────
def _sleeve(cfg: Config, loader: ParquetDataLoader, ramp: Ramp, probe: bool
            ) -> Optional[SatelliteBacktesterV2]:
    """사테라이트 슬리브를 만든다. ramp=None + probe=False 면 None(원본 V1 슬리브를 쓰게).

    프로브와 대조군을 **같은 인자**로 만들어야 대조가 의미 있으므로 조립을 한곳에 모은다.
    """
    if ramp is None and not probe:
        return None                       # 슬리브 미주입 → IRPBacktesterV2 가 원본 V1 을 쓴다
    kw = {}
    if ramp is not None:
        lo, full, floor = ramp
        kw = dict(ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    cls = ExposureProbe if probe else SatelliteBacktesterV2
    return cls(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost, **kw)


def _run(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader, ramp: Ramp,
         probe: bool, start, end, allow_missing: bool):
    """IRP 백테스트 1회. probe=True 면 노출을 관측하는 슬리브를 끼운다."""
    sat = _sleeve(cfg, loader, ramp, probe)
    bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                         satellite=sat, allow_missing=allow_missing)
    res = bt.run(icfg, start=start, end=end)
    exp = sat.exposure(icfg.satellite_weight) if isinstance(sat, ExposureProbe) else None
    return res, exp


def _measure(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader, ramp: Ramp,
             label: str, start, end, allow_missing: bool) -> ExposureResult:
    """노출을 재고, 프로브가 결과를 흔들지 않았음을 대조군으로 검증한다.

    Raises:
        RuntimeError: 프로브 유무로 자산곡선이 달라진 경우(= 측정이 대상을 바꿈).
    """
    res, exp = _run(cfg, icfg, loader, ramp, True, start, end, allow_missing)
    ref, _ = _run(cfg, icfg, loader, ramp, False, start, end, allow_missing)
    if not res.equity.equals(ref.equity):
        gap = (res.equity - ref.equity).abs().max()
        # V1(ramp=None)에서는 대조군이 원본 SatelliteBacktester 이고 프로브는 그 축퇴형이라,
        # 여기서 갈리면 원인이 둘 중 하나다: (a) 프로브가 판정을 건드림, (b) 축퇴 보장 붕괴
        # (원본과 V2 축퇴가 더는 비트 일치하지 않음). 어느 쪽이든 노출 수치를 믿을 수 없다.
        cause = ("프로브가 판정을 건드렸거나 원본↔V2 축퇴 일치가 깨졌습니다"
                 if ramp is None else "관측이 대상을 바꿨습니다")
        raise RuntimeError(
            f"[{label}] 프로브를 끼운 곡선이 대조군과 다릅니다(최대 격차 {gap:.3e}). "
            f"{cause} — 이 노출 수치는 쓸 수 없습니다.")
    logger.info(f"[{label}] 프로브 무해 검증 통과 — 대조군과 자산곡선 완전 일치")
    return exp


# ── 출력 ────────────────────────────────────────────────────────
def _log_summary(v2: ExposureResult, v1: ExposureResult, gate_v1: float) -> None:
    """공식 지표(실효 노출률)와 이진 환산 참고치를 로그로."""
    full = v2.sleeve_weight * 100.0
    logger.info("")
    logger.info("[공식 지표 — 실효 노출률(동결 V2)]")
    logger.info(f"  체크 {len(v2.fill)}회 · 만충 = 포트폴리오 {full:.0f}%")
    logger.info(f"  평균 실효 노출: {v2.portfolio_exposure.mean() * 100:>5.1f}% "
                f"(만충 대비 {v2.fill.mean() * 100:.1f}%)")
    logger.info(f"  중앙값        : {v2.portfolio_exposure.median() * 100:>5.1f}%")
    logger.info(f"  최저          : {v2.portfolio_exposure.min() * 100:>5.1f}% "
                f"({v2.portfolio_exposure.idxmin():%Y-%m})")
    logger.info("  상태 분포:")
    for state, n in v2.shortfall_cause.value_counts().items():
        logger.info(f"    {state:<28} {n:>3}회")

    logger.info("")
    logger.info(f"[참고치 — V1 기준(이진 게이트 {gate_v1:.0f}). 비교용이며 상품 수치가 아니다]")
    short = (v1.slots_used < v1.top_n).sum()
    logger.info(f"  슬롯 미달: {short}/{len(v1.slots_used)}개월")
    logger.info(f"  평균 실효 노출: {v1.portfolio_exposure.mean() * 100:>5.1f}%")
    for state, n in v1.shortfall_cause.value_counts().items():
        logger.info(f"    {state:<28} {n:>3}회")


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP 실효 노출률 분석(동결 V2 기준)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="유니버스 종목이 빠져도 진행(기본은 중단). 빠진 채로 나온 수치는 "
                         "제안서에 쓰지 말 것.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    lo, full, floor = FROZEN_RAMP
    gate_v1 = icfg.satellite.entry_score
    logger.info(f"실효 노출률 분석 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    v2 = _measure(cfg, icfg, loader, FROZEN_RAMP, "동결 V2", start, end, args.allow_missing)
    v1 = _measure(cfg, icfg, loader, None, "V1 기준", start, end, args.allow_missing)
    _log_summary(v2, v1, gate_v1)

    rep = ExposureReport(args.out)
    rep.write_monthly(v2, v1)
    rep.plot_exposure(v2, v1,
                      v2_label=f"동결 V2 · 경사 {lo:.0f}→{full:.0f}(바닥 {floor:.0%})",
                      v1_label=f"V1 기준 · 이진 게이트 {gate_v1:.0f}")


if __name__ == "__main__":
    main()
