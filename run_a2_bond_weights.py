"""A-2 채권 슬리브 월별 배분 내보내기(읽기 전용 전시물) — 판정·커밋 없음, 숫자만.

A-2 spec(채권 스위칭, 153130/114260/439870)의 백테스트에서 **매 체크 채권 슬리브가 듀레이션을
어떻게 나눴나**를 CSV 로 낸다. 관측은 `BondSleeveProbe`(= `HoldingsProbe` 상속)로, 엔진의 선정·
경사 판정을 재구현하지 않고 그대로 받아 적는다. 프로브 유무로 슬리브 자산곡선이 달라지지 않음을
매 실행 대조해(무해 검증) 통과해야 값을 신뢰한다.

채권 cfg 는 `BondSwitchBacktester._bond_sleeve_cfg` 에서 그대로 가져와 A-2 spec 이 실제로 쓰는
슬리브(문턱 52·경사 52→60·바닥 0.3·ramp_hold·top_n=3·현금=153130·주기 M)와 동일하게 맞춘다.

라벨: ① KOSPI200(069500) 국면(200일선 위/아래)과 60일 수익률. ② 금리 방향 **보조** 라벨 2종 —
국고10년 직접 시계열이 없어 채권 가격변화의 역으로 대용(가격↓=금리↑): 장기=국고30(439870) ·
중기=국고3(114260). 439870 은 2022-08 상장이라 그 이전 체크는 점수 NaN·비중 0·장기금리 '미상'
(창 절단 방어), 반면 114260 은 2016~ 전체 이력이라 2020~2022 급등기까지 커버한다. 보조 라벨은
슬리브 신호와 겹치는 순환 성격이 있어 참고용이다.

실행:
    uv run python run_a2_bond_weights.py                     # 전체 기간(A-2 창)
    uv run python run_a2_bond_weights.py --end 2025-12-31    # 기간 컷
산출물:
    reports/a2_bond_weights_monthly.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2
from v3.bond_holdings import BondSleeveProbe
from v3.bond_switch import BondSwitchBacktester

logger = logging.getLogger("run_a2_bond_weights")

_BOND_UNIVERSE = ["153130", "114260", "439870"]
_CASH_TICKER = "153130"                      # 채권 슬리브 현금 대용 = 단기채(금리 급등기 회귀처)
_SHORT_NAMES = {"153130": "단기채", "114260": "국고3", "439870": "국고30"}
_KOSPI200 = "069500"                          # 라벨용 국면 지표(KODEX 200)
_MA_WINDOW = 200
_LONG_BOND = "439870"                          # 장기금리 방향 대용(국고30 가격의 역)
_MID_BOND = "114260"                           # 중기금리 방향 대용(국고3 가격의 역 · 2016~ 전체 이력)
_RATE_WINDOW = 60                              # 방향 판정 창(거래일)
_COL_LONG = "장기금리방향(국고30역)"           # 라벨 컬럼명(엇갈림 필터에서 재사용)
_COL_MID = "중기금리방향(국고3역)"


# ── 조립 ────────────────────────────────────────────────────────
def _bond_cfg(icfg: IRPConfig, loader, cost: float):
    """A-2 spec 이 실제로 쓰는 채권 슬리브 cfg 를 그대로 가져온다(재정의 금지)."""
    bsw = BondSwitchBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cost,
                               satellite=None, bond_universe=list(_BOND_UNIVERSE),
                               cash_ticker=_CASH_TICKER)
    return bsw._bond_sleeve_cfg(icfg)


def _sleeve(loader, cost: float, probe: bool):
    """채권 슬리브 백테스터. probe=True 면 배분을 관측하는 프로브로(같은 경사 인자)."""
    lo, full, floor = FROZEN_RAMP
    kw = dict(ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    cls = BondSleeveProbe if probe else SatelliteBacktesterV2
    return cls(loader=loader, indicator=TrendScoreIndicator(), cost=cost, **kw)


def _measure(cfg: Config, icfg: IRPConfig, loader, start, end) -> pd.DataFrame:
    """채권 배분을 관측하고, 프로브가 슬리브 자산곡선을 흔들지 않았음을 대조군으로 검증한다."""
    bcfg = _bond_cfg(icfg, loader, cfg.cost)

    probe = _sleeve(loader, cfg.cost, True)
    res = probe.run(bcfg, start=start, end=end, trailing=None)
    ref = _sleeve(loader, cfg.cost, False).run(bcfg, start=start, end=end, trailing=None)
    if not res.equity.equals(ref.equity):
        gap = (res.equity - ref.equity).abs().max()
        raise RuntimeError(f"프로브 곡선이 대조군과 다릅니다(최대 격차 {gap:.3e}) — 관측이 대상을 "
                           f"바꿨습니다. 이 배분 수치는 쓸 수 없습니다.")
    logger.info(f"프로브 무해 검증 통과 — 채권 슬리브 자산곡선 완전 일치 · "
                f"{res.equity.index[0]:%Y-%m-%d}~{res.equity.index[-1]:%Y-%m-%d}")
    return probe.bond_weights(_BOND_UNIVERSE, _CASH_TICKER, _SHORT_NAMES)


# ── 라벨(KOSPI200 국면) ──────────────────────────────────────────
def _attach_label(df: pd.DataFrame, loader) -> pd.DataFrame:
    """체크일에 매크로 라벨을 붙인다(전용 · 판정 아님).

    · KOSPI200(069500) 국면(200일선 위/아래)과 60일 수익률.
    · 금리 방향 **보조** 라벨 2종: 국고10년 직접 시계열이 없어 채권 60일 가격변화의 **역**으로
      대용한다(가격↓ = 금리↑). 장기=국고30(439870, 2022-08~ → 그 이전 '미상') · 중기=국고3
      (114260, 2016~ 전체 이력 → 2020~2022 급등기 커버). A-2 슬리브가 참조하는 신호와 겹치는
      순환 성격이 있어 독립 매크로가 아닌 **참고용 보조** 라벨이다.
    """
    checks = pd.to_datetime(df["체크일"])
    df = df.copy()

    # KOSPI200 국면
    close = loader.load(_KOSPI200).df["close"]
    ma = close.rolling(_MA_WINDOW).mean()
    ret60 = close.pct_change(_RATE_WINDOW, fill_method=None)
    c = close.reindex(checks, method="ffill").to_numpy()
    m = ma.reindex(checks, method="ffill").to_numpy()
    r = ret60.reindex(checks, method="ffill").to_numpy()
    df["KOSPI200_국면"] = np.where(np.isnan(m), "미상", np.where(c >= m, "상승", "하락"))
    df["KOSPI200_60일%"] = np.round(r * 100, 2)

    # 금리 방향(보조) — 채권 가격변화의 역(가격↓=금리↑). 상장/워밍업 전은 NaN → '미상'.
    #   장기: 국고30(439870, 2022-08~ → 2020~2022 초는 미상) · 중기: 국고3(114260, 2016~ 전체 이력).
    for tag, ticker in ((_COL_LONG, _LONG_BOND), (_COL_MID, _MID_BOND)):
        dp = (loader.load(ticker).df["close"].pct_change(_RATE_WINDOW, fill_method=None)
              .reindex(checks, method="ffill").to_numpy())
        df[tag] = np.where(np.isnan(dp), "미상", np.where(dp < 0, "상승", "하락"))
        df[f"{'국고30' if ticker == _LONG_BOND else '국고3'}_60일가격%"] = np.round(dp * 100, 2)
    return df


def _divergence(df: pd.DataFrame) -> pd.DataFrame:
    """장기(국고30)·중기(국고3) 금리 방향이 엇갈린 체크만 추린다(둘 다 유효 + 서로 다름).

    장기 라벨이 '미상'(439870 상장 전)인 구간은 비교 불가라 제외된다 → 사실상 2022-08 이후만 대상.
    """
    both_valid = (df[_COL_LONG] != "미상") & (df[_COL_MID] != "미상")
    return df[both_valid & (df[_COL_LONG] != df[_COL_MID])].copy()


def _dir_vs_inclusion(df: pd.DataFrame, dir_col: str, short: str, ticker: str
                      ) -> Tuple[pd.DataFrame, int]:
    """금리 방향 × 슬리브 편입 여부(비중>0) 대조표 + 제외한 '미상' 체크 수.

    '금리 하락(=채권 가격 상승) 국면에 슬리브가 그 채권을 더 담나'를 방향별 편입률·평균비중으로
    본다(판정 아님 — 빈도 사실만). 방향은 60일 가격변화 기준(백워드), 편입은 그 체크의 목표비중.
    '미상'(국고30 상장 전 등)은 비중이 구조적으로 0 이라 행동 대조를 오염시키므로 제외한다.
    """
    wcol = f"{short}({ticker})_비중%"
    valid = df[df[dir_col] != "미상"]
    g = valid.assign(_편입=valid[wcol] > 0).groupby(dir_col)
    tab = pd.DataFrame({
        "편입(n)": g["_편입"].sum().astype(int),
        "미편입(n)": g["_편입"].apply(lambda s: int((~s).sum())),
        "합계(n)": g.size(),
        "편입률%": (g["_편입"].mean() * 100).round(1),
        f"평균{short}비중%": g[wcol].mean().round(2),
    })
    tab.index.name = f"{short}금리방향"
    return tab.reset_index(), int((df[dir_col] == "미상").sum())


def _log_summary(df: pd.DataFrame) -> None:
    """표 개요 — 평균 듀레이션 배분·439870 편입 시작 체크."""
    logger.info("")
    logger.info("[A-2 채권 배분 개요]")
    logger.info(f"  체크 {len(df)}회 · {df['체크일'].iloc[0]} ~ {df['체크일'].iloc[-1]}")
    for t in _BOND_UNIVERSE:
        col = f"{_SHORT_NAMES[t]}({t})_비중%"
        logger.info(f"  {_SHORT_NAMES[t]}({t}) 평균 비중 {df[col].mean():.1f}% "
                    f"(최저 {df[col].min():.1f} · 최고 {df[col].max():.1f})")
    active = df[df[f"국고30(439870)_비중%"] > 0]
    if len(active):
        logger.info(f"  439870(국고30) 첫 편입 체크: {active['체크일'].iloc[0]} "
                    f"(상장 2022-08 이전은 비중 0·점수 NaN)")


def main() -> None:
    ap = argparse.ArgumentParser(description="A-2 채권 슬리브 월별 배분 내보내기(읽기 전용)")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    logger.info(f"A-2 채권 슬리브 배분 · 구간 {start} ~ {end} · 동결 경사 {FROZEN_RAMP}")

    df = _measure(cfg, icfg, loader, start, end)
    df = _attach_label(df, loader)
    _log_summary(df)
    writer = ReportWriter(args.out)
    writer._write_csv(df, "a2_bond_weights_monthly", index=False)

    # 장기/중기 금리 방향이 엇갈린 체크만 별도 파일로(스티프닝·플래트닝 구간 점검용).
    div = _divergence(df)
    both = ((df[_COL_LONG] != "미상") & (df[_COL_MID] != "미상")).sum()
    logger.info(f"장기/중기 금리 방향 엇갈림: {len(div)}/{both}회(둘 다 유효 중)")
    writer._write_csv(div, "a2_bond_divergence", index=False)

    # 금리 방향 × 편입 여부 대조표(국고3·국고30 각각). '미상'은 제외하고 그 수만 로그로 남긴다.
    for dir_col, short, ticker, fname in (
            (_COL_MID, "국고3", _MID_BOND, "a2_bond3_dir_vs_inclusion"),
            (_COL_LONG, "국고30", _LONG_BOND, "a2_bond30_dir_vs_inclusion")):
        tab, n_unknown = _dir_vs_inclusion(df, dir_col, short, ticker)
        wcol, dcol = f"평균{short}비중%", f"{short}금리방향"
        logger.info("")
        logger.info(f"[{short} 금리 방향 × {short} 편입 대조] (미상 {n_unknown}회 제외)")
        for _, r in tab.iterrows():
            logger.info(f"  {r[dcol]:<4} 편입 {r['편입(n)']:>2} · 미편입 {r['미편입(n)']:>2} "
                        f"· 합 {r['합계(n)']:>2} · 편입률 {r['편입률%']:>5.1f}% · 평균비중 {r[wcol]:>5.2f}%")
        writer._write_csv(tab, fname, index=False)


if __name__ == "__main__":
    main()
