"""원안 37종 × Tier 2-a 측정 — 2×2 Calmar 분해의 빈 칸을 채운다(읽기 전용, 동결 무수정).

동결 V2 는 유니버스 36종(금 411060 편입) + Tier 2-a 다. 2025컷 Calmar 분해에서 2×2 중 한 칸이
비어 있다:

              37종        36종
  원설계      0.77   →    0.68
  Tier 2-a     ???   →    1.09   ← 이 스크립트가 채운다

이 러너는 **파라미터를 탐색하지 않는다.** 문턱 52·경사 바닥 30%·만충 60 을 동결값 그대로 두고,
바뀌는 것은 유니버스뿐이다(37종: 411060 대신 죽은 티커 0072R0·0189B0 → 실효 35 live). 네 칸을
모두 재측정해 문서에 기록된 세 칸(0.77/0.68/1.09)을 재현하고, 그 재현이 맞을 때만 빈 칸을 신뢰한다.

**동결 무수정.** config/irp.json·frozen 커밋을 건드리지 않는다. 37종 유니버스는 로드한 설정을
deepcopy 해 메모리에서만 만든다(원본 설정 객체 불변). 판정(채택/기각)은 하지 않는다 — 숫자만 낸다.

실행:
    uv run python run_universe37.py
산출물:
    reports/universe37_tier2a.csv   4구성 × 2구간 전지표 + plateau 그리드
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from irp.config import _DEFAULT_UNIVERSE
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_universe37")

_CUT = "2025-12-31"    # 2025년 말 컷(동결 규율: 전체 + 2025컷 두 구간 모두 확인)


@dataclass(frozen=True)
class Metrics:
    """한 구성·한 구간의 성과 지표 묶음."""
    cagr: float
    sharpe: float
    mdd: float
    calmar: float
    worst_year_pct: float
    worst_year: int
    y2022: Optional[float]


# ── 유니버스 조립 ────────────────────────────────────────────────
def _make_universe37(u36: List[str]) -> List[str]:
    """36종(411060 포함)에서 원안 37종을 만든다: 411060 자리에 0072R0·0189B0(둘 다 죽은 티커).

    35 core 는 두 유니버스가 공유하므로, 36↔37 의 유일한 live 차이는 411060(금 현물) 하나다.
    결과가 irp/config.py 의 원안 기본 유니버스와 정확히 같은지 대조해 전사(轉寫) 오류를 막는다.
    """
    u37: List[str] = []
    for t in u36:
        if t == "411060":
            u37 += ["0072R0", "0189B0"]
        else:
            u37.append(t)
    if u37 != list(_DEFAULT_UNIVERSE):
        raise RuntimeError(
            "[universe37] 구성한 37종이 irp/config.py 원안(_DEFAULT_UNIVERSE)과 다릅니다 — "
            f"측정 대상이 원안이 아닙니다.\n  구성={u37}\n  원안={list(_DEFAULT_UNIVERSE)}")
    return u37


def _icfg_with(icfg: IRPConfig, universe: List[str]) -> IRPConfig:
    """유니버스만 갈아끼운 IRP 설정 사본(원본 설정 객체는 불변)."""
    c = copy.deepcopy(icfg)
    c.satellite.universe = list(universe)
    return c


def _sleeve(cfg: Config, loader: ParquetDataLoader, threshold: Optional[float]):
    """Tier 2-a 슬리브. threshold=None 이면 None(원설계 V1 이진 게이트 60 을 쓰게 한다).

    threshold 는 신규 자격 문턱 겸 경사 하단(ramp_score)이다. 동결은 52. 경사 바닥 0.3·만충 60 고정.
    """
    if threshold is None:
        return None
    _, full, floor = FROZEN_RAMP
    return SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                 ramp_score=threshold, full_score=full, ramp_floor=floor,
                                 ramp_hold=True)


def _measure(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader, universe: List[str],
             threshold: Optional[float], start: str, end: Optional[str]) -> Metrics:
    """한 구성(유니버스 × 문턱)·한 구간을 백테스트해 지표를 뽑는다."""
    sat = _sleeve(cfg, loader, threshold)
    bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                         satellite=sat)
    res = bt.run(_icfg_with(icfg, universe), start=start, end=end)
    m = res.metrics["strategy"]
    worst = min(res.yearly(), key=lambda r: r["strat_pct"])
    y2022 = next((r["strat_pct"] for r in res.yearly() if r["year"] == 2022), None)
    return Metrics(cagr=m["cagr_pct"], sharpe=m["sharpe"], mdd=m["mdd_pct"], calmar=m["calmar"],
                   worst_year_pct=worst["strat_pct"], worst_year=int(worst["year"]), y2022=y2022)


# ── 출력 ────────────────────────────────────────────────────────
def _log_2x2(cells: dict) -> None:
    """2×2 Calmar 표(전체·2025컷)를 로그로. cells[(uni, strat)] = (full, cut)."""
    for span, i in (("전체 Calmar", 0), ("2025컷 Calmar", 1)):
        logger.info("")
        logger.info(f"[{span}]  (→ 는 411060 편입: 37종→36종)")
        logger.info(f"  {'':<8}{'37종':>9}{'36종':>9}{'유니버스효과':>12}")
        for strat in ("원설계", "Tier2a"):
            c37 = cells[("37", strat)][i]
            c36 = cells[("36", strat)][i]
            logger.info(f"  {strat:<8}{c37:>9.3f}{c36:>9.3f}{c36 - c37:>+12.3f}")
        e_ws = cells[("36", "원설계")][i] - cells[("37", "원설계")][i]
        pred = cells[("36", "Tier2a")][i] - e_ws           # 독립 가정 시 37종 Tier2a 예측
        actual = cells[("37", "Tier2a")][i]
        logger.info(f"  독립가정 예측(37종 Tier2a) = {cells[('36','Tier2a')][i]:.3f} − ({e_ws:+.3f})"
                    f" = {pred:.3f} · 실측 {actual:.3f} · 격차 {actual - pred:+.3f}")


def _log_8numbers(t37: Metrics, t37c: Metrics, frozen: Metrics, frozenc: Metrics,
                  plateau: List[tuple]) -> None:
    """37종+Tier2a 8개 수치를 동결 V2(36종+Tier2a)와 나란히."""
    logger.info("")
    logger.info("[8개 수치 — 37종+Tier2a vs 동결 V2(36종+Tier2a)]")
    logger.info(f"  {'지표':<18}{'37종+Tier2a':>14}{'동결 V2(36종)':>16}")
    rows = [
        ("1 전체 CAGR%", t37.cagr, frozen.cagr),
        ("2 전체 Sharpe", t37.sharpe, frozen.sharpe),
        ("3 전체 MDD%", t37.mdd, frozen.mdd),
        ("4 전체 Calmar", t37.calmar, frozen.calmar),
        ("5 2025컷 Calmar", t37c.calmar, frozenc.calmar),
        ("6 2022 수익%", t37.y2022, frozen.y2022),
    ]
    for name, a, b in rows:
        logger.info(f"  {name:<18}{a:>14.2f}{b:>16.2f}")
    logger.info(f"  {'7 최저해 수익%':<18}{t37.worst_year_pct:>10.1f}({t37.worst_year})"
                f"{frozen.worst_year_pct:>12.1f}({frozen.worst_year})")
    logger.info("  8 plateau(문턱 48~52 · 바닥30%·만충60 · 전체/2025컷 Calmar):")
    for thr, full_cal, cut_cal in plateau:
        logger.info(f"      문턱 {thr:>4.0f}:  전체 {full_cal:.3f}  · 2025컷 {cut_cal:.3f}")


# ── 조립 ────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = icfg.start or cfg.start
    u36 = list(icfg.satellite.universe)
    u37 = _make_universe37(u36)
    thr = FROZEN_RAMP[0]     # 동결 문턱 52
    logger.info(f"37종×Tier2a 측정 · 구간 {start} ~ 전체/{_CUT} · 동결 경사 {FROZEN_RAMP} "
                f"· 37종(411060 미편입, 0072R0·0189B0 죽은티커) live≈35")

    # 2×2: (유니버스 37/36) × (전략 원설계=None / Tier2a=52). 각 칸 (전체, 2025컷) 전지표.
    grid = {"37": u37, "36": u36}
    strat = {"원설계": None, "Tier2a": thr}
    full_m: dict = {}
    cut_m: dict = {}
    for uk, uni in grid.items():
        for sk, th in strat.items():
            full_m[(uk, sk)] = _measure(cfg, icfg, loader, uni, th, start, None)
            cut_m[(uk, sk)] = _measure(cfg, icfg, loader, uni, th, start, _CUT)

    cells = {k: (full_m[k].calmar, cut_m[k].calmar) for k in full_m}
    _log_2x2(cells)

    # plateau: 37종에서 문턱만 48·50·52 로 흔든다(바닥30%·만충60 고정). 그 범위 밖(≥55·<48)은 미검증.
    plateau = []
    for t in (48.0, 50.0, 52.0):
        fm = _measure(cfg, icfg, loader, u37, t, start, None)
        cm = _measure(cfg, icfg, loader, u37, t, start, _CUT)
        plateau.append((t, fm.calmar, cm.calmar))

    _log_8numbers(full_m[("37", "Tier2a")], cut_m[("37", "Tier2a")],
                  full_m[("36", "Tier2a")], cut_m[("36", "Tier2a")], plateau)

    # CSV: 4구성 × 2구간 전지표 + plateau 그리드.
    rows = []
    for uk in ("37", "36"):
        for sk in ("원설계", "Tier2a"):
            for span, mm in (("전체", full_m[(uk, sk)]), ("2025컷", cut_m[(uk, sk)])):
                rows.append({
                    "유니버스": f"{uk}종", "전략": sk, "구간": span,
                    "CAGR%": round(mm.cagr, 2), "Sharpe": round(mm.sharpe, 3),
                    "MDD%": round(mm.mdd, 2), "Calmar": round(mm.calmar, 3),
                    "2022%": round(mm.y2022, 2) if mm.y2022 is not None else None,
                    "최저해%": round(mm.worst_year_pct, 2), "최저해": mm.worst_year,
                })
    for t, fc, cc in plateau:
        rows.append({"유니버스": "37종", "전략": f"plateau문턱{t:.0f}", "구간": "전체/2025컷",
                     "Calmar": round(fc, 3), "MDD%": None,
                     "2025컷Calmar": round(cc, 3)})
    ReportWriter("reports")._write_csv(pd.DataFrame(rows), "universe37_tier2a", index=False)


if __name__ == "__main__":
    main()
