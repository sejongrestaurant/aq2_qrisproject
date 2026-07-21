"""V3 탐색 배터리 러너 — 동결 V2 위 변형 실험 전수 실행(원본·V2·설정 무수정).

전달문 배터리를 한 번에 돌려 실험별 표준 보고(두 구간 지표표 + 연도별 매트릭스 + 관문 4개
통과 여부 + 예상 실패 모드 실현 여부)를 내고, 마지막에 종합표 1장을 CSV 로 쓴다. **판정은
하지 않는다** — 관문은 산술 조건일 뿐이고 채택/기각은 사람이 숫자만 보고 한다.

절대 제약(전달문): config/irp.json·동결 커밋·원본 파일 무수정. 모든 변형은 로드한 설정을
deepcopy 해 메모리에서만 만든다. 한 번에 하나(1차에서 조합 금지).

실행:
    uv run python run_v3.py
산출물:
    reports/v3_battery_summary.csv   실험 × 두 구간 전지표 + 관문 통과 여부
"""
from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import pandas as pd

from analysis.report_base import ReportWriter
from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2
from satellite.trailing import AtrTrailingStop
from v3.bond_switch import BondSwitchBacktester
from v3.measure import CUT, Metrics, metrics_of, report, summary_row
from v3.overlay import vol_target
from v3.regime_bond import RegimeBondBacktester
from v3.irp_trailing import TrailingSleeveV2
from v3.regional_cap import (RegionalCapBacktester, REGION_GROUPS, BROAD_REGION_GROUPS,
                             GEO_3GROUPS_EX_REAL, REAL_ASSET_TICKERS)

logger = logging.getLogger("run_v3")


# ── 공통 조립 ────────────────────────────────────────────────────
def frozen_sleeve(cfg: Config, loader: ParquetDataLoader) -> SatelliteBacktesterV2:
    """동결 Tier 2-a 슬리브(문턱 52·경사 52→60·바닥 0.3·ramp_hold)."""
    lo, full, floor = FROZEN_RAMP
    return SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                 ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)


def frozen_bt(cfg: Config, loader: ParquetDataLoader) -> IRPBacktesterV2:
    """동결 V2 기준선 백테스터."""
    return IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                           satellite=frozen_sleeve(cfg, loader))


def cash_returns(loader: ParquetDataLoader, index: pd.DatetimeIndex,
                 ticker: str = "153130") -> pd.Series:
    """현금 대용(단기채) 일간수익을 날짜축에 맞춰 만든다(오버레이 감속 몫 수익용)."""
    close = loader.load(ticker).df["close"].reindex(index).ffill()
    return close.pct_change(fill_method=None).fillna(0.0)


def _two_windows(bt, icfg: IRPConfig, start, end_full) -> Tuple[Metrics, Metrics]:
    """한 백테스터를 전체·2025컷 두 구간으로 재서 Metrics 두 개를 낸다."""
    full = metrics_of(bt.run(copy.deepcopy(icfg), start=start, end=end_full))
    cut = metrics_of(bt.run(copy.deepcopy(icfg), start=start, end=CUT))
    return full, cut


# ── 실험 [A-1] 정적 장기채 ───────────────────────────────────────
def exp_a1(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """채권 114260(국고3) → 439870(국고30) 정적 교체. 대피처 153130 불변.

    439870 이 2022-08 상장이라 IRP 공통 거래일이 절단된다 → 전체 창 관문은 측정 불가.
    같은 절단 창에서 동결 기준선을 나란히 재서 head-to-head 로만 본다.
    """
    icfg_a1 = copy.deepcopy(icfg)
    icfg_a1.bonds = {"153130": 0.10, "439870": 0.10, "273130": 0.10}

    bt = frozen_bt(cfg, loader)
    # A-1 을 기본 start 로 돌려 실제 절단 시작일을 알아낸 뒤, 기준선을 같은 창으로 맞춘다.
    a1_full = metrics_of(bt.run(copy.deepcopy(icfg_a1), start=start, end=end_full))
    trunc_start = a1_full.span.split("~")[0]
    a1_cut = metrics_of(bt.run(copy.deepcopy(icfg_a1), start=trunc_start, end=CUT))

    base = frozen_bt(cfg, loader)
    b_full = metrics_of(base.run(copy.deepcopy(icfg), start=trunc_start, end=end_full))
    b_cut = metrics_of(base.run(copy.deepcopy(icfg), start=trunc_start, end=CUT))

    report("[A-1] 정적 장기채 114260→439870(국고30) · 절단 창", a1_full, a1_cut,
           expected_failure=(f"439870 상장 2022-08 → 창이 {trunc_start} 로 절단. 2022 H1 금리 "
                             "급등기(A-1 의 핵심 논거)가 데이터에 아예 없음 → 관문 측정 불가."),
           base_full=b_full, base_cut=b_cut)
    r = summary_row("A-1 정적 장기채(절단창)", a1_full, a1_cut)
    r["비고"] = f"창절단 {trunc_start}~ · 관문 측정불가"
    rows.append(r)


# ── 실험 [A-2] 채권 추세 스위칭 ──────────────────────────────────
def exp_a2(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """채권 슬리브를 동결 규칙으로 로테이션. 두 구성을 함께 잰다:

    (spec) 153130/114260/439870 — 전달문 지정. 439870 은 2022-08 상장이라 그 이전엔 후보에서
        빠질 뿐(슬리브가 상장 늦음을 흡수) → 전체 창(2020~) 측정 가능. 단 273130(종합채권)이
        빠지고 439870 이 들어와 '동적 vs 정적' 과 '채권 구성 변화' 가 섞인다(불순 비교).
    (iso) 153130/114260/273130 — 동결과 **같은 3 채권**을 정적 고정 대신 동적 로테이션.
        구성 변화 없이 '스위칭' 효과만 분리한다(순수 isolation).
    """
    def make(universe):
        return BondSwitchBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                    satellite=frozen_sleeve(cfg, loader), bond_universe=universe)

    for tag, uni, note in (
        ("spec 439870", ["153130", "114260", "439870"], "273130→439870·동적(불순)"),
        ("iso 동결3채권", ["153130", "114260", "273130"], "동결 3채권 동적화(순수 isolation)"),
    ):
        full, cut = _two_windows(make(uni), icfg, start, end_full)
        report(f"[A-2] 채권 추세 스위칭 · {tag}({'/'.join(uni)})", full, cut,
               expected_failure=("금리 급등기 장기채 점수↓ → 단기채(153130) 자동 회귀. "
                                 "spec 은 2022 H1 에 439870 이 아직 없어 그 회귀는 153130↔114260 "
                                 "안에서만 일어남."))
        r = summary_row(f"A-2 채권스위칭 {tag}", full, cut)
        r["비고"] = note
        rows.append(r)


# ── 실험 [A-2 plateau] 스위칭 손잡이 주변값(채택 자격 검증) ───────
def exp_a2_plateau(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """A-2 spec 의 4/4 통과가 단일 지점 운인지 구조(plateau)인지 판별.

    spec 구성(153130/114260/439870) 고정, **스위칭 손잡이** 두 축의 주변값을 각각 재서 관문 4개가
    면 전체에서 유지되는지만 표로 낸다(판정 없음). 손잡이:
      · 문턱(ramp_score): 채권 슬리브 경사 하단. 동결 52 → 주변 48/52/56(만충 60·바닥 0.3 고정).
      · 주기(check_period): 채권 로테이션 점검 주기. 동결 M → 주변 M/Q.
    한 축을 재는 동안 다른 축은 spec 값에 고정하는 십자(star) 측정 — center(52,M)=spec 재현.
    """
    def make(ramp, period):
        return BondSwitchBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                    satellite=frozen_sleeve(cfg, loader),
                                    bond_universe=["153130", "114260", "439870"],
                                    bond_ramp_score=ramp, bond_check_period=period)

    # (문턱, 주기, 라벨) — center 는 한 번만. 문턱축 3점 + 주기축 M/Q(center 공유).
    points = [(48, "M"), (52, "M"), (56, "M"), (52, "Q")]
    logger.info("")
    logger.info("════ [A-2 plateau] 채권 스위칭 손잡이 주변값(spec 구성 · 관문 4개) ════")
    logger.info(f"  {'문턱':<6}{'주기':<6}{'전체Calmar':>11}{'컷Calmar':>10}{'최저해%':>9}"
                f"{'2022%':>8}{'관문통과':>9}")
    for ramp, period in points:
        full, cut = _two_windows(make(ramp, period), icfg, start, end_full)
        r = summary_row(f"A-2plateau 문턱{ramp}/{period}", full, cut)
        mark = " ← spec" if (ramp, period) == (52, "M") else ""
        r["비고"] = f"A-2 plateau · 스위칭 손잡이{mark}"
        rows.append(r)
        logger.info(f"  {ramp:<6}{period:<6}{full.calmar:>11.3f}{cut.calmar:>10.3f}"
                    f"{full.worst_year_pct:>9.1f}"
                    f"{(full.y2022 if full.y2022 is not None else float('nan')):>8.2f}"
                    f"{r['관문통과수']:>7}/4{mark}")


# ── 실험 [B-1] 변동성 타깃 오버레이 ──────────────────────────────
def exp_b1(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """포트 60일 실현변동성이 목표(10%/12%) 초과 시 위험노출 비례 축소."""
    for target in (0.10, 0.12):
        base = frozen_bt(cfg, loader)
        res_full = base.run(copy.deepcopy(icfg), start=start, end=end_full)
        res_cut = base.run(copy.deepcopy(icfg), start=start, end=CUT)
        cr_full = cash_returns(loader, res_full.equity.index)
        cr_cut = cash_returns(loader, res_cut.equity.index)
        m_full = metrics_of(vol_target(res_full, cr_full, target))
        m_cut = metrics_of(vol_target(res_cut, cr_cut, target))
        report(f"[B-1] 변동성 타깃 {target * 100:.0f}% · 창 60일", m_full, m_cut,
               expected_failure=("게이트와 이중 감속 → 반등 초입 상승을 두 번 깎아 "
                                 "CAGR·Calmar 동반 하락 여부."))
        rows.append(summary_row(f"B-1 변동성타깃 {target * 100:.0f}%", m_full, m_cut))


# ── 실험 [C-1] 트레일링 스탑 활성화 ──────────────────────────────
def exp_c1(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """동결 슬리브 + ATR(×2.5, 14) 트레일링 스탑(config stops 기구현값)."""
    lo, full, floor = FROZEN_RAMP
    def make():
        sleeve = TrailingSleeveV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                  trailing=AtrTrailingStop(atr_period=14, mult=2.5),
                                  ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
        return IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                               satellite=sleeve)
    m_full, m_cut = _two_windows(make(), icfg, start, end_full)
    report("[C-1] 트레일링 스탑 ATR×2.5 활성화(IRP on)", m_full, m_cut,
           expected_failure=("게이트와 기능 중복 → 추세장 조기 청산으로 상승만 깎고 "
                             "MDD 개선은 미미(게이트가 이미 방어)."))
    rows.append(summary_row("C-1 트레일링스탑 ATR×2.5", m_full, m_cut))


# ── 실험 [C-2] 국면 연동 채권비중 ────────────────────────────────
def exp_c2(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """KOSPI200 200MA 하락장 판정 시 채권 30→50%."""
    def make():
        return RegimeBondBacktester(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                    satellite=frozen_sleeve(cfg, loader),
                                    bear_bond_weight=0.50, regime_ticker="069500", ma_window=200)
    m_full, m_cut = _two_windows(make(), icfg, start, end_full)
    report("[C-2] 국면 연동 채권비중 30→50%(KOSPI200 200MA)", m_full, m_cut,
           expected_failure=("2020·2023·2025 V자 반등 초입에 채권을 늘려 재진입을 더 늦추면 "
                             "최저 해·CAGR 손실. 2022 방어는 강화 기대."))
    rows.append(summary_row("C-2 국면연동 채권 30→50%", m_full, m_cut))


# ── 실험 [C-3] 지역 집중 상한 ────────────────────────────────────
def exp_c3(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """지역 집중 상한 — 두 그룹핑 대조. 동결 슬리브에 그룹 상한만 얹어 Top-7 지역 독식을 막는다.

    · 세부 4그룹(글로벌9·미국섹터9·원자재리츠3·한국섹터15): 상한 2/3/4(2×4=8≥7 충족).
    · 광역 3지역(미국9·글로벌12·한국15 — 실물→글로벌): 상한 3/4/5(3×3=9≥7 충족).
    · 세부 3그룹(글로벌9·미국9·한국15 — 실물 3종 상한 면제): 상한 3/4/5. 광역3 과 같은 3지역·
      같은 cap 이라 **실물을 글로벌에 합치느냐 vs 면제하느냐만 다른** 통제 비교가 된다.
      cap 2 는 6<7 로 강제 미충전이라 제외(광역3·세부3 공통).

    상한이 top_n(7) 이상이면 축퇴(동결 기준선과 비트일치). 상한값은 단일 채택이 아니라 면(plateau)
    으로 보고, 각 상한을 관문 4개로 재 숫자만 낸다(판정 없음).
    """
    lo, full_s, floor = FROZEN_RAMP

    def run_scheme(tag: str, groups, caps, note: str, uncapped=None) -> None:
        logger.info("")
        logger.info(f"════ [C-3] 지역 집중 상한 · {tag}({'·'.join(groups)}) 각 상한 {'/'.join(map(str, caps))} ════")
        for cap in caps:
            sleeve = RegionalCapBacktester(loader=loader, indicator=TrendScoreIndicator(),
                                           cost=cfg.cost, cap=cap, groups=groups, uncapped=uncapped,
                                           ramp_score=lo, full_score=full_s,
                                           ramp_floor=floor, ramp_hold=True)
            bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                 satellite=sleeve)
            m_full, m_cut = _two_windows(bt, icfg, start, end_full)
            report(f"[C-3] {tag} 상한 = 그룹당 {cap}슬롯", m_full, m_cut,
                   expected_failure=("상한이 조이면 점수 낮은 타 지역이 강제 편입돼 추세장 상승 일부 "
                                     "포기(CAGR·전체 Calmar↓). 분산 이득이 그 대가를 넘는지가 관건."))
            r = summary_row(f"C-3 {tag} 그룹당{cap}슬롯", m_full, m_cut)
            r["비고"] = note
            rows.append(r)

    run_scheme("세부4그룹", REGION_GROUPS, (2, 3, 4), "지역 집중 상한(세부 4그룹)")
    run_scheme("광역3지역", BROAD_REGION_GROUPS, (3, 4, 5), "지역 집중 상한(광역 3지역·실물→글로벌)")
    run_scheme("세부3그룹", GEO_3GROUPS_EX_REAL, (3, 4, 5), "지역 집중 상한(세부 3그룹·실물 면제)",
               uncapped=REAL_ASSET_TICKERS)


# ── 실험 [E] top_n 강건성(관문 제외 · 문서용) ────────────────────
def exp_e(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """top_n 3/5/7/10 대조 — '왜 7인가' 방어 자료. 관문 판정 대상 아님."""
    lo, full, floor = FROZEN_RAMP
    logger.info("")
    logger.info("════ [E] top_n 강건성(관문 제외 · '왜 7인가' 문서용) ════")
    logger.info(f"  {'top_n':<8}{'전체Calmar':>10}{'컷Calmar':>10}{'전체CAGR%':>10}"
                f"{'전체MDD%':>9}{'최저해%':>9}{'2022%':>8}")
    for tn in (3, 5, 7, 10):
        sleeve = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                       ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
        bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                             satellite=sleeve)
        ic = copy.deepcopy(icfg)
        ic.satellite.top_n = tn
        mf = metrics_of(bt.run(copy.deepcopy(ic), start=start, end=end_full))
        mc = metrics_of(bt.run(copy.deepcopy(ic), start=start, end=CUT))
        mark = " ← 동결" if tn == 7 else ""
        logger.info(f"  {tn:<8}{mf.calmar:>10.3f}{mc.calmar:>10.3f}{mf.cagr:>10.1f}"
                    f"{mf.mdd:>9.1f}{mf.worst_year_pct:>9.1f}"
                    f"{(mf.y2022 if mf.y2022 is not None else float('nan')):>8.2f}{mark}")
        r = summary_row(f"E top_n={tn}(문서용)", mf, mc)
        r["비고"] = "관문 제외 · 강건성 문서용"
        rows.append(r)


# ── 실험 [F] 비용 민감도(관문 제외 · 문서용) ─────────────────────
def exp_f(cfg, icfg, loader, start, end_full, rows: List[dict]) -> None:
    """왕복비용 0.10/0.20/0.30% 대조 — 동결 V2 의 비용 강건성. 관문 판정 대상 아님.

    비용은 백테스터 생성자 인자(float)만 바꿔 조립한다(config·원본 무수정). 0.10% 는 현행
    동결값이라 기준선과 비트 단위로 재현되어야 한다(자기검증). 방어(MDD)는 비용과 무관하고
    수익만 완만히 감쇄하는지를 본다.
    """
    lo, full, floor = FROZEN_RAMP
    logger.info("")
    logger.info("════ [F] 비용 민감도(관문 제외 · 동결 V2 비용 강건성) ════")
    logger.info(f"  {'왕복비용%':<9}{'전체CAGR%':>10}{'전체MDD%':>9}{'전체Calmar':>11}"
                f"{'컷Calmar':>10}")
    for cost in (0.001, 0.002, 0.003):
        sleeve = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cost,
                                       ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
        bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cost,
                             satellite=sleeve)
        mf = metrics_of(bt.run(copy.deepcopy(icfg), start=start, end=end_full))
        mc = metrics_of(bt.run(copy.deepcopy(icfg), start=start, end=CUT))
        mark = " ← 동결" if abs(cost - 0.001) < 1e-9 else ""
        logger.info(f"  {cost * 100:<9.2f}{mf.cagr:>10.1f}{mf.mdd:>9.1f}{mf.calmar:>11.3f}"
                    f"{mc.calmar:>10.3f}{mark}")
        r = summary_row(f"F 비용={cost * 100:.2f}%(문서용)", mf, mc)
        r["비고"] = "관문 제외 · 비용민감도 문서용"
        rows.append(r)


# ── 조립 ────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    start = icfg.start or cfg.start
    end_full = icfg.end or cfg.end

    # 기준선(동결 V2)을 종합표 맨 위에 — 재현 확인은 run_v2.py 와 일치해야 한다.
    base = frozen_bt(cfg, loader)
    bf, bc = _two_windows(base, icfg, start, end_full)
    logger.info(f"동결 V2 기준선 재현 · 전체 Calmar {bf.calmar:.3f} / 컷 {bc.calmar:.3f} "
                f"/ 최저해 {bf.worst_year_pct:.1f} / 2022 {bf.y2022:.2f}")

    rows: List[dict] = [{**summary_row("기준선(동결 V2)", bf, bc), "비고": "기준선"}]

    exp_a1(cfg, icfg, loader, start, end_full, rows)
    exp_a2(cfg, icfg, loader, start, end_full, rows)
    exp_a2_plateau(cfg, icfg, loader, start, end_full, rows)
    exp_b1(cfg, icfg, loader, start, end_full, rows)
    exp_c1(cfg, icfg, loader, start, end_full, rows)
    exp_c2(cfg, icfg, loader, start, end_full, rows)
    exp_c3(cfg, icfg, loader, start, end_full, rows)
    exp_e(cfg, icfg, loader, start, end_full, rows)
    exp_f(cfg, icfg, loader, start, end_full, rows)

    ReportWriter("reports")._write_csv(pd.DataFrame(rows), "v3_battery_summary", index=False)
    logger.info("")
    logger.info("[종합표] reports/v3_battery_summary.csv 저장 완료")


if __name__ == "__main__":
    main()
