"""구간 상태별 손익 러너 — 동결 V2 로테이션의 승률·손익비·PF·슬롯대별 평균(제안서 재료).

동결 V2(경사 52→60·바닥 30%) 사테라이트 슬리브를 돌려 `rotations_log` 를 얻고, 구간을 슬롯
수로 분류해 집계한다. 특히 진단서가 지목한 '전환 구간(5슬롯대)'이 부분 진입 도입 후 어떻게
바뀌었는지를 V1 기록과 대비할 수 있게 낸다.

실행:
    uv run python run_segments.py                     # 전체 기간
    uv run python run_segments.py --end 2025-12-31    # 기간 컷
산출물:
    reports/segment_state.csv   슬롯대별 (구간 수 · 평균 구간수익%)
"""
from __future__ import annotations

import argparse
import logging

from analysis.report_base import ReportWriter
from analysis.segments import compute_segment_stats
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_segments")


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP 구간 상태별 손익(동결 V2 기준)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
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
    top_n = icfg.satellite.top_n
    logger.info(f"구간 상태별 손익 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    sat = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    res = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                          satellite=sat).run(icfg, start=start, end=end)

    stats = compute_segment_stats(res.rotations_log or [])
    logger.info("")
    for line in stats.summary_lines(top_n):
        logger.info(line)

    # CSV: 슬롯대별 표에 '만충 대비 상태' 라벨을 붙여 저장.
    out = stats.by_slot.copy()
    out.insert(0, "상태", ["현금방어" if n == 0 else ("만충" if n == top_n else "전환/방어")
                          for n in out.index])
    out.index.name = "슬롯수"
    ReportWriter(args.out)._write_csv(out, "segment_state", index=True)


if __name__ == "__main__":
    main()
