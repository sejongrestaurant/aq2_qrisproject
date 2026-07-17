"""월별 구성(보유 종목) 러너 — 동결 V2 의 체크 시점별 보유 상세를 CSV 로 낸다(제안서 전시물).

`run_exposure.py` 가 내는 '체크별 총노출' 아래 한 단계를 편다: 각 체크에서 어느 종목을 몇 점·
몇 % 로 들었고 다음 달까지 각자·슬리브가 얼마를 벌었는지를 행 단위로 기록한다.

실행:
    uv run python run_holdings.py                     # 전체 기간
    uv run python run_holdings.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    holdings_monthly.csv   체크일 × 보유 종목. 컬럼: 체크일·다음체크일·코드·종목명·TrendScore·
        충전율%·포트폴리오비중%·종목구간수익%·슬리브구간수익%·위험자산총노출%.
        (뒤 세 컬럼은 체크 단위 합계로, 같은 체크의 모든 종목 행에 반복된다.)

**측정 신뢰성 — 프로브가 결과를 흔들지 않음을 매 실행 검증한다.** 관측용 프로브를 끼운
백테스트와 끼우지 않은 대조군의 자산곡선을 대조해, 다르면 멈춘다(`run_exposure.py` 와 동일한
무해 검증). 이게 통과해야 "이 구성은 동결 상품이 실제로 쓴 값"이라고 말할 수 있다.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from analysis.holdings import HoldingsProbe
from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_holdings")


# ── 조립 ────────────────────────────────────────────────────────
def _sleeve(cfg: Config, loader: ParquetDataLoader, probe: bool):
    """동결 V2 사테라이트 슬리브를 만든다. probe=True 면 보유 구성을 관측하는 프로브로.

    프로브와 대조군을 **같은 경사 인자**로 만들어야 대조가 의미 있으므로 조립을 한곳에 모은다.
    """
    lo, full, floor = FROZEN_RAMP
    kw = dict(ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    cls = HoldingsProbe if probe else SatelliteBacktesterV2
    return cls(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost, **kw)


def _run(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader, probe: bool,
         start, end, allow_missing: bool):
    """IRP 백테스트 1회. probe=True 면 보유 구성을 관측하는 슬리브를 끼운다."""
    sat = _sleeve(cfg, loader, probe)
    bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                         satellite=sat, allow_missing=allow_missing)
    res = bt.run(icfg, start=start, end=end)
    return res, sat


def _measure(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader,
             start, end, allow_missing: bool) -> pd.DataFrame:
    """보유 구성을 관측하고, 프로브가 자산곡선을 흔들지 않았음을 대조군으로 검증한다.

    Raises:
        RuntimeError: 프로브 유무로 자산곡선이 달라진 경우(= 관측이 대상을 바꿈).
    """
    res, sat = _run(cfg, icfg, loader, True, start, end, allow_missing)
    ref, _ = _run(cfg, icfg, loader, False, start, end, allow_missing)
    if not res.equity.equals(ref.equity):
        gap = (res.equity - ref.equity).abs().max()
        raise RuntimeError(
            f"프로브를 끼운 곡선이 대조군과 다릅니다(최대 격차 {gap:.3e}). "
            f"관측이 대상을 바꿨습니다 — 이 보유 구성 수치는 쓸 수 없습니다.")
    logger.info("프로브 무해 검증 통과 — 대조군과 자산곡선 완전 일치")
    return sat.holdings(icfg.satellite_weight, icfg.satellite.names)


# ── 출력 ────────────────────────────────────────────────────────
def _log_summary(df: pd.DataFrame) -> None:
    """행 수·체크 수·평균 노출 등 표의 개요를 로그로."""
    checks = df["체크일"].nunique()
    held = df[df["코드"] != "(현금)"]
    per_check = df.groupby("체크일")["위험자산총노출%"].first()
    logger.info("")
    logger.info("[보유 구성 개요 — 동결 V2]")
    logger.info(f"  행 {len(df)}개 · 체크 {checks}회 · 보유 종목행 {len(held)}개")
    logger.info(f"  체크당 평균 보유 종목수: {len(held) / checks:.1f}")
    logger.info(f"  평균 위험자산 총노출: {per_check.mean():.1f}% "
                f"(최저 {per_check.min():.1f}% · 최고 {per_check.max():.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="IRP 월별 보유 구성 내보내기(동결 V2 기준)")
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
    logger.info(f"월별 보유 구성 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    df = _measure(cfg, icfg, loader, start, end, args.allow_missing)
    _log_summary(df)
    ReportWriter(args.out)._write_csv(df, "holdings_monthly", index=False)


if __name__ == "__main__":
    main()
