"""롤링 창 원자료 내보내기 — 창 × 시작월 × 상품 × 수익률(거치/적립 병기).

**왜 원자료인가.** 지금까지의 산출물은 집계(최저·중앙값·손실 확률)였다. 집계는 "어느 쪽이
낫나"에는 답하지만 "**언제 시작한 사람이** 그랬나"에는 답하지 못한다. 이 CSV 는 창 하나
하나를 그대로 내보내, 받는 쪽에서 원하는 방식으로(피벗·분포·특정 시기 필터) 다시 볼 수 있게 한다.

**거치와 적립을 한 파일에 담되 지표를 섞지 않는다.** 두 방식은 분모가 다르다:

  · 거치(일시납) — 시작 시점에 한 번 넣고 N개월 보유. 수익률 = **연율 수익률(%)**.
  · 적립(월 납입 등) — 창 안에서 계속 넣는다. 수익률 = **총 납입액 대비 손익률(%)** 로,
    돈이 평균적으로 절반 기간만 투자돼 있어 거치식과 **같은 축에서 비교하면 안 된다**.
    그 비교가 필요할 때 쓰라고 금액가중수익률(MWR, 연율)을 별도 열로 함께 낸다.

그래서 `구분`·`지표` 열을 반드시 함께 읽어야 한다. 두 구분을 섞어 평균 내면 분모가 다른
수치가 한 숫자로 뭉개진다.

실행:
    uv run python run_rolling_export.py                     # 전체 기간
    uv run python run_rolling_export.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    rolling_windows.csv   구분 · 납입계획 · 보유개월 · 시작월 · 상품 · 수익률% · MWR연율% · 지표
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Sequence

import pandas as pd

from analysis.cashflow import AnnualLump, CashflowPlan, MonthlyDCA
from analysis.frozen import build_irp
from analysis.report_base import ReportWriter
from analysis.rolling import RollingAnalyzer
from analysis.rolling_returns import RollingReturns
from config import Config
from data import ParquetDataLoader
from irp import IRPConfig
from run_v2 import FROZEN_RAMP

logger = logging.getLogger("run_rolling_export")

# 보유기간(개월). 78개월 구간이라 60개월 창은 18개뿐 — 소표본임을 CSV 를 받는 쪽도 알도록
# 창 수를 로그로 함께 낸다.
HORIZONS: Sequence[int] = (12, 24, 36, 60)

_LUMP = "일시납(거치)"
_M_LUMP = "연율수익률%"
_M_DCA = "납입액대비손익률%"


# ── 수집 ────────────────────────────────────────────────────────
def _lump_rows(curves: Dict[str, pd.Series], horizons: Sequence[int]) -> List[dict]:
    """거치식 창 원자료 — 시작월마다 (상품, 연율 수익률)."""
    roll = RollingReturns(curves)
    rows: List[dict] = []
    for h in horizons:
        w = roll.windows(h)
        for start, r in w.iterrows():
            for label in curves:
                rows.append({
                    "구분": "거치", "납입계획": _LUMP, "보유개월": h,
                    "시작월": f"{start:%Y-%m}", "상품": label,
                    "수익률%": round(float(r[label]), 3), "MWR연율%": None, "지표": _M_LUMP,
                })
    return rows


def _dca_rows(curves: Dict[str, pd.Series], plans: Sequence[CashflowPlan],
              horizons: Sequence[int]) -> List[dict]:
    """적립식 창 원자료 — 시작월마다 (상품, 납입계획, 납입액 대비 손익률, MWR)."""
    rows: List[dict] = []
    for label, eq in curves.items():
        ra = RollingAnalyzer(eq)
        for plan in plans:
            for h in horizons:
                df = ra.distribution_frame(plan, h)
                for start, r in df.iterrows():
                    rows.append({
                        "구분": "적립", "납입계획": plan.name, "보유개월": h,
                        "시작월": f"{start:%Y-%m}", "상품": label,
                        "수익률%": round(float(r["profit_pct"]), 3),
                        "MWR연율%": (None if pd.isna(r["mwr_pct"])
                                    else round(float(r["mwr_pct"]), 3)),
                        "지표": _M_DCA,
                    })
    return rows


# ── 출력 ────────────────────────────────────────────────────────
def _log_shape(df: pd.DataFrame) -> None:
    """받는 쪽이 표본 크기를 오해하지 않게 창 수를 구분별로 찍는다."""
    logger.info("")
    logger.info(f"[내보낸 창 수 — 총 {len(df):,}행]")
    logger.info(f"{'구분':<6}{'납입계획':<18}{'보유':>6}{'상품별 창수':>12}")
    g = (df.groupby(["구분", "납입계획", "보유개월", "상품"], sort=False).size()
         .groupby(level=[0, 1, 2]).max())
    for (kind, plan, h), n in g.items():
        logger.info(f"{kind:<6}{plan:<18}{h:>4}개{n:>12}")
    logger.info("  · 창끼리 구간이 겹쳐 독립 표본이 아니다(보유가 길수록 창이 급감).")
    logger.info("  · 거치와 적립은 분모가 다르다 — '구분'·'지표' 열을 함께 읽을 것.")


def main() -> None:
    ap = argparse.ArgumentParser(description="롤링 창 원자료 내보내기(거치·적립 병기)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    ap.add_argument("--allow-missing", action="store_true", help="유니버스 결손 허용(기본 중단).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    logger.info(f"롤링 창 내보내기 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    res = build_irp(loader, cfg.cost, FROZEN_RAMP,
                    allow_missing=args.allow_missing).run(icfg, start=start, end=end)
    curves = {"HELM(동결 V2)": res.equity, res.benchmark_name: res.benchmark}
    plans: List[CashflowPlan] = [MonthlyDCA(), AnnualLump()]

    rows = _lump_rows(curves, HORIZONS) + _dca_rows(curves, plans, HORIZONS)
    df = pd.DataFrame(rows, columns=["구분", "납입계획", "보유개월", "시작월", "상품",
                                     "수익률%", "MWR연율%", "지표"])
    _log_shape(df)
    ReportWriter(args.out)._write_csv(df, "rolling_windows")


if __name__ == "__main__":
    main()
