"""국면별 성과 분해 러너 — 상승·하락·횡보에서 V2 · V1 · TRF7030 이 어떻게 갈리나.

"하락 방어형"이라는 상품 주장을 검증하는 표다. 전체 구간 한 줄로는 방어를 증명할 수 없다 —
하락장에서 벤치마크보다 덜 잃고, 그 대가로 상승장에서 뒤처지는 **교환**이 실제로 있었는지를
국면을 갈라 봐야 한다. 동결 V2 의 국면별 평균 노출을 함께 실어 그 인과(게이트가 노출을
줄여서 덜 잃었다)를 잇는다.

국면은 KOSPI200(069500) 200일 이동평균 + 기울기로 나눈다(`analysis/regime.py`).
**사후 라벨이지 매매 신호가 아니다** — 전략은 이 라벨을 보지 않는다.

V1(이진 게이트 60)을 함께 놓는 이유: 동결 V2 가 겨냥한 것이 '전환 국면의 노출 복원'이므로,
개선이 있었다면 횡보·상승 초입에서 V1 과 벌어져야 한다. 같은 표에서 그 방향이 확인된다.

실행:
    uv run python run_regime.py                     # 전체 기간
    uv run python run_regime.py --end 2025-12-31    # 기간 컷

산출물(reports/):
    regime_performance.csv   국면 × 대상 (일수·비중·누적수익·연율화·MDD·평균노출)
    regime_equity.png        국면 음영 자산곡선 + 하단 실효 노출 패널(발표용)
"""
from __future__ import annotations

import argparse
import logging

from analysis.exposure import ExposureProbe
from analysis.frozen import build_irp
from analysis.regime import classify_regime, compute_regime_table, summary_lines
from analysis.regime_report import RegimeReport
from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from irp import IRPConfig
from run_benchmark import KOSPI200, KOSPI200_NAME
from run_v2 import FROZEN_RAMP

logger = logging.getLogger("run_regime")

_V2 = "HELM(동결 V2)"
_V1 = "V1 기준선(이진 60)"


def main() -> None:
    ap = argparse.ArgumentParser(description="국면별 성과 분해(KOSPI200 200MA 기준)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--ma", type=int, default=200, help="국면 판정 이동평균 길이(기본 200일).")
    ap.add_argument("--slope", type=int, default=20, help="이동평균 기울기 창(기본 20일).")
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
    logger.info(f"국면별 성과 분해 · 구간 {start} ~ {end} · "
                f"{KOSPI200_NAME} {args.ma}MA(기울기 {args.slope}일) 기준")

    # 동결 V2 는 노출 프로브를 끼워 돌린다 — 프로브는 판정에 관여하지 않으므로(관측 전용)
    # 자산곡선은 프로브 없는 실행과 동일하다(무해 검증은 run_exposure.py 가 매 실행 수행).
    probe_kw = dict(sleeve_cls=ExposureProbe, allow_missing=args.allow_missing)
    bt_v2 = build_irp(loader, cfg.cost, FROZEN_RAMP, **probe_kw)
    res_v2 = bt_v2.run(icfg, start=start, end=end)
    exposure = bt_v2.satellite.exposure(icfg.satellite_weight).portfolio_exposure

    res_v1 = build_irp(loader, cfg.cost, None,
                       allow_missing=args.allow_missing).run(icfg, start=start, end=end)

    idx = res_v2.equity.index
    labels = classify_regime(loader.load(KOSPI200).df["close"], idx,
                             ma_window=args.ma, slope_window=args.slope)
    curves = {
        _V2: res_v2.equity,
        _V1: res_v1.equity.reindex(idx).ffill(),
        res_v2.benchmark_name: res_v2.benchmark,
    }
    table = compute_regime_table(curves, labels, exposure=exposure, exposure_for=_V2)

    logger.info("")
    for line in summary_lines(table):
        logger.info(line)
    logger.info("  · MDD 는 그 국면 구간만 이어 붙인 합성곡선 기준 — 전체 구간 MDD 와 정의가 다르다.")
    logger.info(f"  · 평균노출% 는 동결 V2 의 포트폴리오 기준 실효 노출(만충 = "
                f"{icfg.satellite_weight * 100:.0f}%).")

    ReportWriter(args.out)._write_csv(table, "regime_performance")
    # 차트는 표와 같은 곡선·라벨을 그대로 쓴다(재계산 없음). 순서는 색 배정 규약대로
    # 상품(파랑) → 벤치마크(주황) → 원설계(먹색 점선).
    RegimeReport(args.out).plot_regime_equity(
        {_V2: curves[_V2], res_v2.benchmark_name: curves[res_v2.benchmark_name],
         _V1: curves[_V1]},
        labels, exposure=exposure, sleeve_weight=icfg.satellite_weight, highlight=_V2)


if __name__ == "__main__":
    main()
