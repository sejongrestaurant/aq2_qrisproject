"""2026 상반기 V1(이진60) vs V2(동결 램프) 격차 귀속 분해 — 읽기 전용, 판정 없이 표만.

동결 V2 와 V1 의 2026-01~06 슬리브 구간수익을 월별로 나란히 놓아 격차가 어느 달에 집중되는지,
그 달의 보유가 (a) 감량 보유(같은 종목 얕게) 때문인지 (b) 구성 차이(다른 종목) 때문인지 가른다.
관측은 `HoldingsProbe`(엔진 관측 · 재구현 없음), 프로브 무해 검증 통과분만 쓴다.

V1 = ramp_score=None(원본 이진 게이트 60/45 로 축퇴, 비트 일치) · V2 = FROZEN_RAMP(52/60/0.3·ramp_hold).
재현: uv run python run_v2_2026_attrib.py → reports/v2_2026_monthly.csv · v2_2026_holdings_gap.csv
"""
from __future__ import annotations

import logging

import pandas as pd

from analysis.holdings import HoldingsProbe
from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_v2_2026_attrib")

_Y = 2026


def _probe(cfg, loader, sat_cfg, start, end, ramp):
    """슬리브를 프로브로 관측하고 무해 검증(프로브 유무 자산곡선 일치) 후 holdings 반환."""
    kw = {} if ramp is None else dict(ramp_score=ramp[0], full_score=ramp[1],
                                      ramp_floor=ramp[2], ramp_hold=True)
    probe = HoldingsProbe(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost, **kw)
    res = probe.run(sat_cfg, start=start, end=end, trailing=None)
    ref = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(),
                                cost=cfg.cost, **kw).run(sat_cfg, start=start, end=end, trailing=None)
    if not res.equity.equals(ref.equity):
        raise RuntimeError("프로브 무해 검증 실패 — 관측 무효.")
    return probe


def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    """체크일별 슬리브 구간수익(중복 제거) 중 2026 상반기만."""
    m = df.groupby("체크일")["슬리브구간수익%"].first().reset_index()
    m["체크일"] = pd.to_datetime(m["체크일"])
    return m[(m["체크일"].dt.year == _Y) & (m["체크일"].dt.month <= 6)].reset_index(drop=True)


def _held(df: pd.DataFrame, date) -> pd.DataFrame:
    """특정 체크일의 보유(현금 자리채움 제외): 코드·종목명·충전율·종목구간수익."""
    d = df[(pd.to_datetime(df["체크일"]) == date) & (df["코드"] != "(현금)")]
    return d[["코드", "종목명", "충전율%", "종목구간수익%"]].set_index("코드")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start, end = icfg.start or cfg.start, icfg.end or cfg.end
    sat_cfg, names = icfg.satellite, icfg.satellite.names

    p1 = _probe(cfg, loader, sat_cfg, start, end, None)          # V1 이진 60
    p2 = _probe(cfg, loader, sat_cfg, start, end, FROZEN_RAMP)   # V2 동결 램프
    logger.info("프로브 무해 검증 통과 — V1·V2 슬리브 모두")

    h1 = p1.holdings(icfg.satellite_weight, names)
    h2 = p2.holdings(icfg.satellite_weight, names)

    # (1) 월별 슬리브 구간수익 V1 vs V2 + 격차.
    m1, m2 = _monthly(h1), _monthly(h2)
    mon = m1.merge(m2, on="체크일", suffixes=("_V1", "_V2"))
    mon["격차%p(V2-V1)"] = (mon["슬리브구간수익%_V2"] - mon["슬리브구간수익%_V1"]).round(2)
    mon["체크일"] = mon["체크일"].dt.strftime("%Y-%m")
    logger.info("")
    logger.info("[2026 상반기 월별 슬리브 구간수익 · V1 vs V2]")
    logger.info(f"  {'월':<9}{'V1%':>9}{'V2%':>9}{'격차%p':>9}")
    for _, r in mon.iterrows():
        logger.info(f"  {r['체크일']:<9}{r['슬리브구간수익%_V1']:>9.2f}"
                    f"{r['슬리브구간수익%_V2']:>9.2f}{r['격차%p(V2-V1)']:>9.2f}")
    logger.info(f"  {'합(단순)':<9}{mon['슬리브구간수익%_V1'].sum():>9.2f}"
                f"{mon['슬리브구간수익%_V2'].sum():>9.2f}{mon['격차%p(V2-V1)'].sum():>9.2f}")
    ReportWriter("reports")._write_csv(mon, "v2_2026_monthly", index=False)

    # (2) 격차 최대 달의 보유 대조.
    worst = mon.loc[mon["격차%p(V2-V1)"].idxmin(), "체크일"]
    wdate = pd.to_datetime(worst + "-01")
    # 그 달의 실제 체크일(월초 거래일)로 매칭.
    d1 = _held(h1, pd.to_datetime(m1[m1["체크일"].dt.strftime("%Y-%m") == worst]["체크일"].iloc[0]))
    d2 = _held(h2, pd.to_datetime(m2[m2["체크일"].dt.strftime("%Y-%m") == worst]["체크일"].iloc[0]))

    codes = sorted(set(d1.index) | set(d2.index))
    rows = []
    for c in codes:
        in1, in2 = c in d1.index, c in d2.index
        nm = (d1.loc[c, "종목명"] if in1 else d2.loc[c, "종목명"])
        kind = "공통" if (in1 and in2) else ("V1만" if in1 else "V2만")
        rows.append({
            "구분": kind, "코드": c, "종목명": nm,
            "V1충전율%": d1.loc[c, "충전율%"] if in1 else 0.0,
            "V2충전율%": d2.loc[c, "충전율%"] if in2 else 0.0,
            "V1종목수익%": d1.loc[c, "종목구간수익%"] if in1 else None,
            "V2종목수익%": d2.loc[c, "종목구간수익%"] if in2 else None,
        })
    gap = pd.DataFrame(rows).sort_values(["구분", "코드"])
    logger.info("")
    logger.info(f"[격차 최대 달 {worst} 보유 대조] (구분 · 충전율 · 종목구간수익)")
    for _, r in gap.iterrows():
        logger.info(f"  {r['구분']:<5}{r['코드']:<8}{str(r['종목명'])[:16]:<17}"
                    f"V1충 {r['V1충전율%']:>5.0f}% · V2충 {r['V2충전율%']:>5.0f}%")
    logger.info(f"  종목수: 공통 {(gap['구분']=='공통').sum()} · V1만 {(gap['구분']=='V1만').sum()} "
                f"· V2만 {(gap['구분']=='V2만').sum()}")
    ReportWriter("reports")._write_csv(gap, "v2_2026_holdings_gap", index=False)


if __name__ == "__main__":
    main()
