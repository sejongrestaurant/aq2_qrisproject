"""구리(160580) 편입 대조 — 세 구성에서 편입 횟수·진입상태·연도별 기여(읽기 전용, 동결 무수정).

목적: 동결 V2 에서 구리 n=15(기여 −2.8%)가 (A) 계측 정상화의 결과인지 (B) 문턱 52 인하가 새로
끌어들인 것인지 판정할 **재료**를 낸다(판정은 하지 않는다 — 숫자만).

방법론은 실효 노출률 프로브와 같다(`HoldingsProbe`): 엔진이 정한 선정·비중·점수를 받아 적기만
하므로 재구현 오차가 없다. 세 구성에서 각각 구리 편입을 센다:
  - V1        : 36종 · 이진 게이트 60         (원설계, ramp_score=None 축퇴)
  - 동결 V2   : 36종 · 문턱 52 경사(바닥30%/만충60)
  - 37종+Tier2a: 원안 37종(411060 미편입) · 문턱 52 경사

각 구성마다 함께 낸다: 총 편입/총 체크 · 진입상태 분해(점수 <52 유지 / 52~60 부분충전 / ≥60 만충) ·
(동결 V2 한정) 구리 기여(−2.8%)의 연도별 분해 — rotations_log 방식(contribution_analysis 와 동일,
n=15 로 재현).

**무해 검증**: 각 구성마다 프로브 유무로 자산곡선이 같은지 대조해, 다르면 멈춘다.
실행:
    uv run python run_copper.py
산출물:
    reports/copper_by_config.csv     3구성 × (편입/체크·상태분해)
    reports/copper_v2_by_year.csv    동결 V2 구리 기여 연도별 분해
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import pandas as pd

from analysis.holdings import HoldingsProbe
from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from irp.config import _DEFAULT_UNIVERSE
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("run_copper")

_COPPER = "160580"


# ── 유니버스 ────────────────────────────────────────────────────
def _universe37(u36: List[str]) -> List[str]:
    """36종에서 원안 37종(411060 자리에 죽은 티커 0072R0·0189B0). 원안과 일치 검증."""
    u37: List[str] = []
    for t in u36:
        u37 += ["0072R0", "0189B0"] if t == "411060" else [t]
    if u37 != list(_DEFAULT_UNIVERSE):
        raise RuntimeError("[copper] 구성한 37종이 irp/config.py 원안과 다릅니다.")
    return u37


def _icfg_with(icfg: IRPConfig, universe: List[str]) -> IRPConfig:
    c = copy.deepcopy(icfg)
    c.satellite.universe = list(universe)
    return c


# ── 프로브 실행 + 무해 검증 ──────────────────────────────────────
def _sleeve(cfg: Config, loader: ParquetDataLoader, ramp: Optional[float], probe: bool):
    """슬리브 하나. ramp=None 이면 V1 이진 게이트(축퇴), 아니면 경사 진입.

    probe=True 면 HoldingsProbe(관측), False 면 SatelliteBacktesterV2(대조군). 같은 인자로 만든다.
    """
    _, full, floor = FROZEN_RAMP
    if ramp is None:
        kw = dict(ramp_score=None)               # V1 이진 게이트로 축퇴(원본과 비트 일치)
    else:
        kw = dict(ramp_score=ramp, full_score=full, ramp_floor=floor, ramp_hold=True)
    cls = HoldingsProbe if probe else SatelliteBacktesterV2
    return cls(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost, **kw)


def _holdings(cfg: Config, icfg: IRPConfig, loader: ParquetDataLoader, universe: List[str],
              ramp: Optional[float], label: str, start, end) -> pd.DataFrame:
    """한 구성의 보유 구성 표(체크×종목)를 관측하고, 프로브 무해성을 대조군으로 검증한다."""
    ic = _icfg_with(icfg, universe)

    def run(probe: bool):
        sat = _sleeve(cfg, loader, ramp, probe)
        res = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                              satellite=sat).run(ic, start=start, end=end)
        return res, sat

    res, sat = run(True)
    ref, _ = run(False)
    if not res.equity.equals(ref.equity):
        gap = (res.equity - ref.equity).abs().max()
        raise RuntimeError(f"[{label}] 프로브가 자산곡선을 흔들었습니다(최대 격차 {gap:.3e}) — "
                           f"구리 편입 수치를 신뢰할 수 없습니다.")
    logger.info(f"[{label}] 무해 검증 통과 · 체크 {res.equity.index[0]:%Y-%m}~{res.equity.index[-1]:%Y-%m}")
    return sat.holdings(icfg.satellite_weight, icfg.satellite.names)


# ── 구리 집계 ────────────────────────────────────────────────────
def _copper_stats(df: pd.DataFrame, label: str) -> Dict:
    """구리 편입을 센다: 총 편입/총 체크 + 점수대 분해(<52 유지 / 52~60 부분 / ≥60 만충).

    총 체크는 **전액 현금 체크까지 포함한 전체 월간 체크 수**(3구성 공통 77)로 둔다 — 구성마다
    현금대피 개월이 달라 분모가 어긋나면 편입률 비교가 오염된다.
    """
    checks = df["체크일"].nunique()
    cu = df[df["코드"] == _COPPER].copy()
    s = cu["TrendScore"]
    below = int((s < 52).sum())          # 유지 구간(보유만, 신규 자격 미달)
    ramp = int(((s >= 52) & (s < 60)).sum())   # 부분충전 구간
    full = int((s >= 60).sum())          # 만충 구간
    partial_fill = int((cu["충전율%"] < 100.0 - 1e-9).sum())   # 실제 부분충전된 슬롯 수
    return {
        "구성": label, "총편입": len(cu), "총체크": checks,
        "점수<52(유지)": below, "점수52~60(부분)": ramp, "점수≥60(만충)": full,
        "충전율<100(부분충전슬롯)": partial_fill,
        "평균TrendScore": round(float(s.mean()), 2) if len(cu) else None,
        "평균충전율%": round(float(cu["충전율%"].mean()), 1) if len(cu) else None,
    }


def _copper_by_year(icfg: IRPConfig, cfg: Config, loader: ParquetDataLoader,
                    u36: List[str], start, end) -> pd.DataFrame:
    """동결 V2 구리 기여(−2.8%)의 연도별 분해 — rotations_log 방식(contribution_analysis 와 동일).

    각 교체 구간에서 구리의 실제 수익과 '동반 보유 대비 초과(vs_peers)'를 구해 연도(구간 시작연도)로
    묶는다. 전체 평균이 −2.8/n=15 로 재현되고, 그 −2.8 이 어느 해에 몰렸는지 드러난다.
    """
    lo, full, floor = FROZEN_RAMP
    sat = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    res = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                          satellite=sat).run(_icfg_with(icfg, u36), start=start, end=end)
    log = res.rotations_log or []
    last_day = res.equity.index[-1]
    closes: Dict[str, pd.Series] = {}

    def seg_ret(code, t0, t1):
        if code not in closes:
            closes[code] = loader.load(code).df["close"]
        s = closes[code].loc[t0:t1]
        return None if len(s) < 2 else float(s.iloc[-1] / s.iloc[0] - 1)

    rows = []
    for i, r in enumerate(log):
        codes = [lb.split("·")[0] for lb in r["labels"]]
        if _COPPER not in codes:
            continue
        t0 = r["date"]
        t1 = log[i + 1]["date"] if i + 1 < len(log) else last_day
        rets = {c: seg_ret(c, t0, t1) for c in codes}
        rets = {c: v for c, v in rets.items() if v is not None}
        if _COPPER not in rets:
            continue
        others = [v for c, v in rets.items() if c != _COPPER]
        rows.append({"연도": t0.year, "구간시작": t0.date(),
                     "구리수익%": round(rets[_COPPER] * 100, 2),
                     "vs_peers%": round((rets[_COPPER] - sum(others) / len(others)) * 100, 2)
                     if others else 0.0})
    df = pd.DataFrame(rows)
    by = df.groupby("연도").agg(구간수=("vs_peers%", "size"),
                              평균구리수익=("구리수익%", "mean"),
                              평균vs_peers=("vs_peers%", "mean")).round(2)
    return df, by


# ── 조립 ────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = icfg.start or cfg.start
    end = icfg.end or cfg.end
    u36 = list(icfg.satellite.universe)
    u37 = _universe37(u36)
    thr = FROZEN_RAMP[0]
    logger.info(f"구리 편입 대조 · 구간 {start} ~ {end or '전체'} · 구리 {_COPPER}")

    configs = [
        ("V1(36종·게이트60)", u36, None),
        ("동결V2(36종·문턱52)", u36, thr),
        ("37종+Tier2a(문턱52)", u37, thr),
    ]
    stats = []
    for label, uni, ramp in configs:
        df = _holdings(cfg, icfg, loader, uni, ramp, label, start, end)
        stats.append(_copper_stats(df, label))

    logger.info("")
    logger.info("[구리(160580) 편입 대조 — 3구성]")
    tbl = pd.DataFrame(stats)
    for line in tbl.to_string(index=False).splitlines():
        logger.info("  " + line)

    # 동결 V2 구리 기여 연도별 분해(−2.8% 가 어느 해인지).
    picks, by_year = _copper_by_year(icfg, cfg, loader, u36, start, end)
    logger.info("")
    logger.info(f"[동결 V2 구리 기여 연도별 분해 · rotations_log 방식 · "
                f"전체 n={len(picks)} 평균vs_peers={picks['vs_peers%'].mean():.2f}%]")
    for line in by_year.to_string().splitlines():
        logger.info("  " + line)

    rep = ReportWriter("reports")
    rep._write_csv(tbl, "copper_by_config", index=False)
    rep._write_csv(by_year.reset_index(), "copper_v2_by_year", index=False)


if __name__ == "__main__":
    main()
