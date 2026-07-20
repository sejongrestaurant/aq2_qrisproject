"""벤치마크 병기 러너 — TRF7030 · KOSPI200 두 잣대 + 추적오차·정보비율(제안서 §8 표).

두 가지를 한 번에 낸다:
  ① 상대지표 — 동결 V2 를 KODEX TRF7030(같은 위험군)과 KOSPI200 069500(투자자의 체감
     잣대) 각각에 대해 재고, 추적오차(액티브의 크기)·정보비율(그 크기의 효율)을 붙인다.
  ② 벤치마크 절대지표 — 두 벤치마크의 CAGR/MDD/Calmar 를 **전체 구간과 2025년 말 컷
     두 구간 모두** 산출한다. 제안서 §8 표의 '벤치 컷 Calmar' 빈칸이 이것이다.

전략 수치를 두 구간에서 재는 규율(전체 + 2025말 컷)을 벤치마크에도 그대로 적용한다 —
한쪽 구간에서만 앞서는 우열 주장은 국면 편중이기 때문이다.

실행:
    uv run python run_benchmark.py                 # 전체 + 2025말 컷 둘 다
    uv run python run_benchmark.py --cut 2024-12-31

산출물(reports/):
    benchmark_absolute.csv   구간 × 대상(전략·TRF7030·KOSPI200) 절대지표
    benchmark_relative.csv   구간 × 벤치마크 상대지표(초과CAGR·TE·IR·상관·베타)
    underwater.png           언더워터(낙폭) 곡선 — 깊이 × 지속(전체 구간, 발표용)
    rolling_returns.csv/png  롤링 수익 분포 — 보유기간별 연율 수익률(거치식, 전체 구간)
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Tuple

import pandas as pd

from analysis.benchmark import BenchmarkComparison, align_curve
from analysis.drawdown_report import DrawdownReport
from analysis.frozen import build_irp
from analysis.report_base import ReportWriter
from analysis.rolling_report import RollingReturnReport
from analysis.rolling_returns import RollingReturns
from config import Config
from data import ParquetDataLoader
from irp import IRPConfig
from run_v2 import FROZEN_RAMP

logger = logging.getLogger("run_benchmark")

# 투자자 체감 잣대. 상품 벤치마크(TRF7030)와 성격이 달라 병기 의미가 있다.
KOSPI200 = "069500"
KOSPI200_NAME = "KODEX 200"

# 롤링 분포 보유기간(개월). 60개월까지만 — 78개월 구간에서 그 이상은 창이 한 자릿수로 준다.
HORIZONS = (12, 24, 36, 60)


def _compare(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader,
             start, end, allow_missing: bool) -> BenchmarkComparison:
    """한 구간의 동결 V2 백테스트 + 두 벤치마크 병기 비교기를 만든다."""
    res = build_irp(loader, cfg.cost, FROZEN_RAMP,
                    allow_missing=allow_missing).run(icfg, start=start, end=end)
    idx = res.equity.index
    # TRF7030 은 엔진이 이미 같은 축으로 정규화해 실어 준다(재계산하면 어긋날 수 있다).
    benches = {
        res.benchmark_name: res.benchmark,
        f"{KOSPI200_NAME}({KOSPI200})": align_curve(loader.load(KOSPI200).df["close"], idx),
    }
    return BenchmarkComparison(res.equity, benches)


def main() -> None:
    ap = argparse.ArgumentParser(description="벤치마크 병기 비교(TRF7030 · KOSPI200)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--cut", default="2025-12-31",
                    help="두 번째 구간의 종료일(기본 2025-12-31). 'none' 이면 전체만.")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="유니버스 결손 허용(기본 중단). 결손 상태 수치는 제안서에 쓰지 말 것.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end

    windows: List[Tuple[str, object]] = [("전체", end)]
    if args.cut.lower() != "none":
        windows.append((f"{args.cut} 컷", args.cut))

    abs_rows, rel_rows = [], []
    full_cmp: BenchmarkComparison | None = None
    for label, w_end in windows:
        logger.info("")
        logger.info(f"── 구간 [{label}] {start} ~ {w_end} ──")
        cmp_ = _compare(cfg, icfg, loader, start, w_end, args.allow_missing)
        for line in cmp_.summary_lines():
            logger.info(line)
        full_cmp = full_cmp or cmp_          # 전시물은 전체 구간(첫 창)으로 그린다
        abs_rows.append(cmp_.absolute().assign(구간=label))
        rel_rows.append(cmp_.relative().assign(구간=label))

    rep = ReportWriter(args.out)
    for rows, name, key in ((abs_rows, "benchmark_absolute", "대상"),
                            (rel_rows, "benchmark_relative", "벤치마크")):
        df = pd.concat(rows, ignore_index=True)
        rep._write_csv(df[["구간", key] + [c for c in df.columns if c not in ("구간", key)]], name)

    # 전시물 2종 — 곡선 순서가 곧 역할(상품 → 상품 벤치마크 → 참고 지수).
    shown = {full_cmp.strategy_label: full_cmp.strategy, **full_cmp.benchmarks}
    DrawdownReport(args.out).plot_underwater(shown)

    roll = RollingReturns(shown)
    logger.info("")
    for line in roll.summary_lines(HORIZONS):
        logger.info(line)
    rep_roll = RollingReturnReport(args.out)
    rep_roll.write_table(roll, HORIZONS)
    rep_roll.plot_box(roll, HORIZONS)


if __name__ == "__main__":
    main()
