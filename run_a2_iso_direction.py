"""A-2 iso 방향성 검증 — iso 3채권 각각의 '금리 방향 × 편입률'을 전체 창에서(읽기 전용).

iso 구성(동결 상품의 실제 채권 3종: 153130 단기채·114260 국고3·273130 종합채권)을 동적 로테이션한
슬리브를 엔진 관측(`BondSleeveProbe`)해, **종합채권(273130)이 금리 하락기에 실제로 더 담기는지**를
방향별 편입률·평균비중으로 확인한다. 성립하면 iso 의 4/4 는 '듀레이션 로테이션'으로 해석되고,
안 하면 다른 효과임을 시사한다(판정 아님 — 빈도 사실만 낸다).

spec(439870 국고30)과 달리 iso 3채권은 모두 2017 이전 상장이라 **절단 없이 전체 창(2020~)** 측정
가능하다 — 그래서 방향성 검증이 spec 보다 iso 에서 더 신뢰된다.

금리 방향 = 각 채권 60일 가격변화의 역(가격↓=금리↑) 대용. 153130(단기채)은 현금 대용 티커라
편입이 거의 상시 100% 라 방향성 신호가 약하다(대조는 273130·114260 이 핵심). 재현:
    uv run python run_a2_iso_direction.py
산출물:
    reports/a2_iso_dir_vs_inclusion.csv   채권 × 금리방향 × 편입률·평균비중
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from satellite.backtester_v2 import SatelliteBacktesterV2
from v3.bond_holdings import BondSleeveProbe
from v3.bond_switch import BondSwitchBacktester
from run_a2_bond_weights import _sleeve, _dir_vs_inclusion, _RATE_WINDOW

logger = logging.getLogger("run_a2_iso_direction")

_ISO_UNIVERSE = ["153130", "114260", "273130"]     # 동결 상품의 실제 채권 3종(동적화)
_CASH_TICKER = "153130"
_SHORT = {"153130": "단기채", "114260": "국고3", "273130": "종합채권"}


def _iso_bond_cfg(icfg: IRPConfig, loader, cost: float):
    """iso 채권 슬리브 cfg(동결 문턱·top_n=3·현금=153130·주기 M) — spec 과 같은 조립 규칙."""
    bsw = BondSwitchBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cost,
                               satellite=None, bond_universe=list(_ISO_UNIVERSE),
                               cash_ticker=_CASH_TICKER)
    return bsw._bond_sleeve_cfg(icfg)


def _measure(cfg: Config, icfg: IRPConfig, loader, start, end) -> pd.DataFrame:
    """iso 채권 배분을 관측하고 프로브 무해 검증 후 wide 표(체크×비중·점수)를 낸다."""
    bcfg = _iso_bond_cfg(icfg, loader, cfg.cost)
    probe = _sleeve(loader, cfg.cost, True)
    res = probe.run(bcfg, start=start, end=end, trailing=None)
    ref = _sleeve(loader, cfg.cost, False).run(bcfg, start=start, end=end, trailing=None)
    if not res.equity.equals(ref.equity):
        gap = (res.equity - ref.equity).abs().max()
        raise RuntimeError(f"프로브 곡선이 대조군과 다릅니다(최대 격차 {gap:.3e}) — 관측 무효.")
    logger.info(f"프로브 무해 검증 통과 · iso 슬리브 "
                f"{res.equity.index[0]:%Y-%m-%d}~{res.equity.index[-1]:%Y-%m-%d}")
    return probe.bond_weights(_ISO_UNIVERSE, _CASH_TICKER, _SHORT)


def _attach_dirs(df: pd.DataFrame, loader) -> pd.DataFrame:
    """각 iso 채권의 60일 가격변화 역(=금리 방향)을 체크일에 붙인다(채권별 자기 가격 기준)."""
    checks = pd.to_datetime(df["체크일"])
    df = df.copy()
    for t in _ISO_UNIVERSE:
        dp = (loader.load(t).df["close"].pct_change(_RATE_WINDOW, fill_method=None)
              .reindex(checks, method="ffill").to_numpy())
        df[f"{_SHORT[t]}금리방향"] = np.where(np.isnan(dp), "미상", np.where(dp < 0, "상승", "하락"))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = icfg.start or cfg.start
    end = icfg.end or cfg.end
    logger.info(f"A-2 iso 방향성 검증 · 구간 {start} ~ {end} · iso {_ISO_UNIVERSE}")

    df = _attach_dirs(_measure(cfg, icfg, loader, start, end), loader)

    # 채권별 금리방향 × 편입 대조를 쌓아 한 표로.
    parts = []
    logger.info("")
    logger.info("[iso 채권별 금리 방향 × 편입 대조 — 전체 창]")
    for t in _ISO_UNIVERSE:
        short = _SHORT[t]
        tab, n_unknown = _dir_vs_inclusion(df, f"{short}금리방향", short, t)
        tab.insert(0, "채권", f"{short}({t})")
        tab = tab.rename(columns={f"{short}금리방향": "금리방향", f"평균{short}비중%": "평균비중%"})
        parts.append(tab)
        note = f" · 미상 {n_unknown}회 제외" if n_unknown else ""
        for _, r in tab.iterrows():
            logger.info(f"  {short:<5} {r['금리방향']:<4} 편입 {r['편입(n)']:>2}/{r['합계(n)']:>2} "
                        f"· 편입률 {r['편입률%']:>5.1f}% · 평균비중 {r['평균비중%']:>5.2f}%{note}")
    out = pd.concat(parts, ignore_index=True)
    ReportWriter("reports")._write_csv(out, "a2_iso_dir_vs_inclusion", index=False)


if __name__ == "__main__":
    main()
