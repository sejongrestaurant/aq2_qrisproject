"""회전율 러너 — 연도별 회전율과 비용 드래그(제안서 '비용은 반영했나' 답변 재료).

두 계층을 따로 잰다. 기준 금액이 다르기 때문이다:
  · 슬리브 로테이션(월간 Top-7 교체) — 사테라이트 슬리브(포트폴리오의 70%) 기준.
  · 상위 리밸런싱(분기 + ±7%p 임계) — 포트폴리오 전체 기준.
슬리브 회전율은 `× satellite_weight` 로 포트폴리오 환산해 두 계층을 더할 수 있게 낸다.

**측정 방식은 관측이지 추정이 아니다.** 목표비중은 점수·가격만의 함수라 포트폴리오 금액과
무관하므로, 비용률만 0 으로 둔 대조 백테스트와의 자산곡선 비율에 엔진이 실제로 청구한
비용 계수 Π(1 − cost×turnover) 가 고스란히 남는다. 여기서 회전율을 되찾는다
(`analysis/turnover.py`). 전제가 깨지면 러너가 아니라 그 모듈이 멈춘다.

실행:
    uv run python run_turnover.py                     # 전체 기간
    uv run python run_turnover.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    turnover_by_year.csv   연도별 계층별 (회전횟수 · 회전율% · 비용드래그%p)
    turnover_summary.csv   계층별 전체·연평균 요약
    turnover_by_year.png   연도별 회전율(계층 누적) + 막대 위 비용 드래그(발표용)
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from analysis.frozen import build_irp, build_sleeve
from analysis.report_base import ReportWriter
from analysis.turnover import TurnoverStats, recover_turnover
from analysis.turnover_report import TurnoverReport
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from run_v2 import FROZEN_RAMP

logger = logging.getLogger("run_turnover")

_SLEEVE = "슬리브 로테이션(월간)"
_TOP = "상위 리밸런싱(분기+임계)"


# ── 측정 ────────────────────────────────────────────────────────
def _sleeve_turnover(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader,
                     start, end) -> TurnoverStats:
    """슬리브 단독 백테스트를 비용 있음/없음으로 돌려 로테이션 회전율을 되찾는다.

    IRP 를 거치지 않고 슬리브만 돌린다 — 상위 리밸런싱 비용이 섞이면 두 계층을 분리할 수 없다.
    """
    curves = {}
    for tag, c in (("cost", cfg.cost), ("free", 0.0)):
        sat = build_sleeve(loader, c, FROZEN_RAMP, indicator=TrendScoreIndicator())
        curves[tag] = sat.run(icfg.satellite, start=start, end=end, trailing=None).equity
    turn = recover_turnover(curves["cost"], curves["free"], cfg.cost, label=_SLEEVE)
    return TurnoverStats(turnover=turn, cost=cfg.cost,
                         scale=icfg.satellite_weight, label=_SLEEVE)


def _top_turnover(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader,
                  start, end, allow_missing: bool) -> TurnoverStats:
    """상위 리밸런싱 회전율 — 슬리브 비용은 **양쪽 모두 켠 채** 상위 비용만 껐다 켠다.

    슬리브 비용까지 함께 끄면 슬리브 수익이 달라져 임계(±7%p) 트리거 날짜가 어긋난다.
    그러면 비교 대상이 '비용만 다른 같은 경로'가 아니게 돼 되찾기 전제가 무너진다.
    """
    curves = {}
    for tag, c in (("cost", cfg.cost), ("free", 0.0)):
        bt = build_irp(loader, c, FROZEN_RAMP, sleeve_cost=cfg.cost, allow_missing=allow_missing)
        curves[tag] = bt.run(icfg, start=start, end=end).equity
    turn = recover_turnover(curves["cost"], curves["free"], cfg.cost, label=_TOP)
    return TurnoverStats(turnover=turn, cost=cfg.cost, scale=1.0, label=_TOP)


# ── 출력 ────────────────────────────────────────────────────────
def _log_summary(stats: list, years: float, cost: float) -> None:
    """계층별 연평균 회전율·비용 드래그와 합계를 로그로."""
    logger.info("")
    logger.info(f"[회전율 · 왕복 거래비용 {cost * 100:.2f}% 기준 · {years:.1f}년]")
    logger.info(f"{'계층':<24}{'횟수':>6}{'총회전율%':>11}{'연평균%':>10}{'연드래그%p':>12}")
    total_ann = 0.0
    for s in stats:
        row = s.summary(years)
        total_ann += row["연평균회전율%"]
        logger.info(f"{row['계층']:<24}{row['회전횟수']:>6}{row['총회전율%']:>11.1f}"
                    f"{row['연평균회전율%']:>10.1f}{row['연평균비용드래그%p']:>12.3f}")
    logger.info(f"{'합계(포트폴리오 기준)':<24}{'':>6}{'':>11}{total_ann:>10.1f}"
                f"{total_ann * cost:>12.3f}")
    logger.info("  · 회전율은 단방향(100% = 포트폴리오를 한 번 갈아엎음).")
    logger.info("  · 슬리브 회전율에는 종목 '교체'뿐 아니라 매달 목표비중으로 되돌리는 "
                "'드리프트 복원'과 부분 충전율 변화가 함께 들어간다(체크 77회 = 매달 발생).")
    logger.info("  · 드래그는 이미 백테스트 수익률에 **반영된** 비용이다(추가로 빼지 말 것).")


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP 회전율·비용 드래그(동결 V2 기준)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="유니버스 결손 허용(기본 중단).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    logger.info(f"회전율 측정 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    sleeve = _sleeve_turnover(cfg, icfg, loader, start, end)
    top = _top_turnover(cfg, icfg, loader, start, end, args.allow_missing)
    stats = [sleeve, top]

    idx = top.turnover.index
    years = max((idx[-1] - idx[0]).days, 1) / 365.25
    _log_summary(stats, years, cfg.cost)

    # 연도별 표: 계층을 열로 펼쳐 한 장에서 비교되게 한다.
    by_year = pd.concat({s.label: s.by_year() for s in stats}, axis=1).fillna(0.0)
    by_year[("합계", "회전율%")] = sum(s.by_year()["회전율%"].reindex(by_year.index).fillna(0.0)
                                    for s in stats)
    by_year[("합계", "비용드래그%p")] = (by_year[("합계", "회전율%")] * cfg.cost).round(3)
    logger.info("")
    logger.info(f"[연도별 회전율%(포트폴리오 기준)]\n{by_year.to_string()}")

    rep = ReportWriter(args.out)
    rep._write_csv(by_year, "turnover_by_year", index=True)
    rep._write_csv(pd.DataFrame([s.summary(years) for s in stats]), "turnover_summary")
    TurnoverReport(args.out).plot_by_year(stats, cfg.cost)


if __name__ == "__main__":
    main()
