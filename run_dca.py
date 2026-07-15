"""적립식 분석 러너 — 동결 V2 곡선 위에 IRP 납입 현금흐름을 얹는다(제안서 §8~9 재료).

전략·체결 엔진은 건드리지 않는다. 동결된 V2(문턱 52 + 52~60 크기 경사·바닥 30%)를 돌려
일간 자산곡선을 얻고, 그 위에 두 납입 방식을 얹어 **투자자가 실제로 받아 가는 숫자**를 낸다.
벤치마크(KODEX TRF7030)에도 같은 현금흐름을 얹어 나란히 낸다.

실행:
    uv run python run_dca.py                     # 전체 구간
    uv run python run_dca.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    dca_summary.csv       전략×납입계획 요약(총 납입액 대비 평가액 · MWR · TWR)
    dca_rolling.csv       보유기간별 롤링 통계(손실 확률 · 중앙값 · 최악)
    dca_loss_curve.png    보유기간별 손실 확률 곡선
    dca_distribution.png  시작 월별 성과 분포
    dca_growth.png        납입 누계 대비 평가액 추이
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List

import pandas as pd

from analysis.cashflow import AnnualLump, CashflowPlan, MonthlyDCA
from analysis.dca import DCAResult, DCASimulator
from analysis.report import DCAReport
from analysis.rolling import HorizonStats, RollingAnalyzer
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_dca")

# 보유기간 격자(개월). 백테스트가 78개월뿐이라 60개월 창은 19개밖에 안 나온다(소표본).
HORIZONS: List[int] = [12, 24, 36, 48, 60]
# 분포 차트를 그릴 대표 보유기간(창 수와 대표성의 절충).
DIST_HORIZON: int = 36


# ── 조립 ────────────────────────────────────────────────────────
def _frozen_curves(cfg: Config, icfg: IRPConfig, start, end,
                   allow_missing: bool = False) -> Dict[str, pd.Series]:
    """동결 V2 전략곡선과 벤치마크곡선을 얻는다({표시명: equity})."""
    lo, full, floor = FROZEN_RAMP
    loader = ParquetDataLoader(cfg.data_dir)
    sat = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    res = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                          satellite=sat, allow_missing=allow_missing
                          ).run(icfg, start=start, end=end)
    # 벤치마크는 IRP 결과에 실려 오는 TRF7030 곡선(시작 1.0 정규화)을 그대로 쓴다.
    return {f"HELM V2(경사 {lo:.0f}→{full:.0f}·{floor:.0%})": res.equity,
            res.benchmark_name: res.benchmark}


def _simulate(curves: Dict[str, pd.Series], plans: List[CashflowPlan]
              ) -> Dict[str, List[DCAResult]]:
    """곡선 × 납입계획 전수 시뮬레이션."""
    out: Dict[str, List[DCAResult]] = {}
    for label, eq in curves.items():
        sim = DCASimulator(eq)
        out[label] = [sim.run(p) for p in plans]
    return out


def _rolling(curves: Dict[str, pd.Series], plans: List[CashflowPlan]
             ) -> Dict[str, List[HorizonStats]]:
    """곡선별 롤링 시작점 통계."""
    return {label: RollingAnalyzer(eq).run(plans, HORIZONS) for label, eq in curves.items()}


# ── 출력 ────────────────────────────────────────────────────────
def _log_summary(results: Dict[str, List[DCAResult]]) -> None:
    """요약표를 로그로(정렬 위해 폭 지정 포맷)."""
    logger.info("")
    logger.info(f"{'전략':<26}{'납입계획':<18}{'납입(만)':>10}{'평가(만)':>10}"
                f"{'손익률%':>9}{'MWR%':>8}{'TWR%':>8}")
    for strat, rs in results.items():
        for r in rs:
            mwr = "  n/a" if r.mwr_pct is None else f"{r.mwr_pct:>8.2f}"
            logger.info(f"{strat:<26}{r.plan_name:<18}{r.contributed / 1e4:>10.0f}"
                        f"{r.final_value / 1e4:>10.0f}{r.profit_pct:>9.2f}{mwr}{r.twr_pct:>8.2f}")


def _log_rolling(stats: Dict[str, List[HorizonStats]]) -> None:
    """보유기간별 손실 확률을 로그로."""
    logger.info("")
    logger.info("[보유기간별 손실 확률 — 모든 시작 월 롤링]")
    logger.info(f"{'전략':<26}{'납입계획':<18}{'보유':>5}{'창수':>6}{'손실확률%':>10}"
                f"{'중앙값%':>9}{'최악%':>9}")
    for strat, ss in stats.items():
        for s in ss:
            logger.info(f"{strat:<26}{s.plan_name:<18}{s.horizon_months:>4}개{s.n_windows:>6}"
                        f"{s.loss_prob_pct:>10.1f}{s.median_pct:>9.2f}{s.worst_pct:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP 적립식 분석(동결 V2 기준)")
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
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    logger.info(f"적립식 분석 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    plans: List[CashflowPlan] = [MonthlyDCA(), AnnualLump()]
    curves = _frozen_curves(cfg, icfg, start, end, allow_missing=args.allow_missing)

    results = _simulate(curves, plans)
    _log_summary(results)
    stats = _rolling(curves, plans)
    _log_rolling(stats)

    rep = DCAReport(args.out)
    rep.write_summary(results)
    rep.write_rolling(stats)
    rep.plot_loss_curve(stats)
    rep.plot_distribution(
        {f"{label} · {plans[0].name}": RollingAnalyzer(eq).distribution(plans[0], DIST_HORIZON)
         for label, eq in curves.items()}, DIST_HORIZON)
    rep.plot_growth({label: DCASimulator(eq).curve(plans[0]) for label, eq in curves.items()})


if __name__ == "__main__":
    main()
