"""A-2 재측정 — 장기 rung(30년/10년/종합) × 중기 rung 그리드. 판정 없이 숫자만.

제3안 확정: KOSEF 국고채10년(148070, 2011-10 상장·실물·IRP 가능)이 실재해, 순수 10년 국채 사다리를
전체 창(2020~)에서 잴 수 있다. **주의**: A-2 성과 측정은 spec(439870 국고30) 도 전체 창이다 —
스위칭이 439870 의 상장 늦음(2022-08)을 흡수하므로(절단되는 것은 A-1 정적 편입뿐). 각 사다리의
측정 span 을 찍어 이를 확인한다.

30년/10년 × 중기물 그리드 + iso 를 나란히 두 구간·관문 4개로 재고, 국고10년의 금리 방향×편입
대조도 낸다(엔진 관측 · 무해 검증).

재현: uv run python run_a2_gov10.py
산출물: reports/a2_gov10_summary.csv · reports/a2_gov10_dir_vs_inclusion.csv
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from run_v2 import FROZEN_RAMP
from run_v3 import frozen_sleeve, frozen_bt, _two_windows
from v3.bond_switch import BondSwitchBacktester
from v3.bond_holdings import BondSleeveProbe
from v3.measure import CUT, metrics_of, report, summary_row
from run_a2_bond_weights import _sleeve, _dir_vs_inclusion, _RATE_WINDOW

logger = logging.getLogger("run_a2_gov10")

_GOV10 = "148070"
_GOV30 = "439870"
_GOV10_LADDER = ["153130", "114260", _GOV10]
# 30년/10년 × 중기물(국고3/종합채권) 그리드 + iso. gov30 = 기존 spec 과 동일 구성(= 30년 사다리).
# 셋 다 스위칭이 상장 늦음을 흡수하므로 전체 창(2020~) 측정 — 각 실행의 span 을 찍어 확인한다.
_LADDERS = {
    "gov30 국고30(153130/114260/439870)": ["153130", "114260", _GOV30],
    "gov30b 종합+국고30(153130/273130/439870)": ["153130", "273130", _GOV30],
    "gov10 국고10(153130/114260/148070)": _GOV10_LADDER,
    "gov10b 종합+국고10(153130/273130/148070)": ["153130", "273130", _GOV10],
    "iso 종합채권(153130/114260/273130)": ["153130", "114260", "273130"],
}


_NAMES = {"153130": "단기채", "114260": "국고3", "273130": "종합채권",
          _GOV10: "국고10", _GOV30: "국고30"}
# 방향성 검증용 장기 rung 3종(중기물은 국고3 으로 통일 → 장기 rung 만 비교).
_LONG_RUNGS = [["153130", "114260", _GOV30],   # 30년
               ["153130", "114260", _GOV10],   # 10년
               ["153130", "114260", "273130"]]  # 종합채권


def _a2(cfg, loader, universe):
    return BondSwitchBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                satellite=frozen_sleeve(cfg, loader), bond_universe=list(universe))


def _long_directional(cfg, loader, icfg, universe, start, end):
    """사다리의 장기 rung(3번째 채권) 금리 방향 × 편입 대조표(엔진 관측 · 무해 검증).

    장기 rung 의 60일 가격변화 역(=금리 방향)으로 그 채권 편입률·평균비중을 방향별로 가른다.
    439870(국고30)은 2022-08 상장이라 그 이전 '미상' 은 제외된다.
    """
    long_t = universe[-1]
    name = _NAMES[long_t]
    bcfg = _a2(cfg, loader, universe)._bond_sleeve_cfg(icfg)
    probe = _sleeve(loader, cfg.cost, True)
    res = probe.run(bcfg, start=start, end=end, trailing=None)
    ref = _sleeve(loader, cfg.cost, False).run(bcfg, start=start, end=end, trailing=None)
    if not res.equity.equals(ref.equity):
        raise RuntimeError(f"프로브 무해 검증 실패 — {name} 사다리 배분 수치 무효.")
    dfw = probe.bond_weights(list(universe), "153130", {t: _NAMES[t] for t in universe})
    checks = pd.to_datetime(dfw["체크일"])
    dp = (loader.load(long_t).df["close"].pct_change(_RATE_WINDOW, fill_method=None)
          .reindex(checks, method="ffill").to_numpy())
    dfw[f"{name}금리방향"] = np.where(np.isnan(dp), "미상", np.where(dp < 0, "상승", "하락"))
    tab, n_unknown = _dir_vs_inclusion(dfw, f"{name}금리방향", name, long_t)
    tab.insert(0, "장기rung", f"{name}({long_t})")
    tab = tab.rename(columns={f"{name}금리방향": "금리방향", f"평균{name}비중%": "평균비중%"})
    return tab, n_unknown


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start, end = icfg.start or cfg.start, icfg.end or cfg.end

    # 기준선(동결 V2) + 세 A-2 사다리 두 구간·관문.
    bf, bc = _two_windows(frozen_bt(cfg, loader), icfg, start, end)
    logger.info(f"기준선(동결 V2) 전체 Calmar {bf.calmar:.3f} / 컷 {bc.calmar:.3f}")
    rows = [{**summary_row("기준선(동결 V2)", bf, bc), "비고": "기준선"}]

    for tag, uni in _LADDERS.items():
        full, cut = _two_windows(_a2(cfg, loader, uni), icfg, start, end)
        logger.info(f"[A-2 재측정] {tag} · span 전체 {full.span} · 컷 {cut.span}")
        r = summary_row(f"A-2 {tag}", full, cut)
        r["비고"] = f"span {full.span}"
        rows.append(r)
    ReportWriter("reports")._write_csv(pd.DataFrame(rows), "a2_gov10_summary", index=False)

    # 그리드 세 장기 rung(30년·10년·종합) 각각 금리 방향 × 편입 대조(iso 처럼).
    logger.info("")
    logger.info("[장기 rung 금리 방향 × 편입 대조 — 중기물 국고3 통일]")
    parts = []
    for uni in _LONG_RUNGS:
        tab, n_unknown = _long_directional(cfg, loader, icfg, uni, start, end)
        parts.append(tab)
        note = f" · 미상 {n_unknown}회 제외" if n_unknown else ""
        for _, r in tab.iterrows():
            logger.info(f"  {r['장기rung']:<14} {r['금리방향']:<4} 편입률 {r['편입률%']:>5.1f}% "
                        f"· 평균비중 {r['평균비중%']:>5.2f}%{note}")
    ReportWriter("reports")._write_csv(pd.concat(parts, ignore_index=True),
                                       "a2_gov_dir_vs_inclusion", index=False)


if __name__ == "__main__":
    main()
