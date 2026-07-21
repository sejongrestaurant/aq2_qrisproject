"""V3 실험 공통 측정 하버스 — 전지표·연도별·관문 4개를 한 곳에서 낸다.

배터리의 모든 실험은 '두 구간(전체·2025컷) 지표표 + 연도별 매트릭스 + 관문 4개 통과 여부 +
예상 실패 모드 실현 여부' 라는 같은 형식으로 보고한다(전달문 규정). 그 공통부를 여기 모아
실험 러너는 '무엇을 바꾸는가'(백테스터 조립)만 정의하면 되게 한다.

**판정하지 않는다.** 관문은 사전 고정한 산술 조건(Calmar·최악해·2022)일 뿐이고, 채택/기각은
사람이 한다. 이 모듈은 통과/미달을 bool 로만 표기한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backtest import BacktestResult

logger = logging.getLogger(__name__)

# 사전 고정 기준선 = 동결 V2(모든 관문·비교의 기준). CLAUDE.md·전달문과 일치.
CUT = "2025-12-31"          # 2025년 말 컷(전체 + 컷 두 구간 확인 규율)
GATE_FULL_CALMAR = 1.05     # 관문 ① 전체 Calmar ≥ 기준선
GATE_CUT_CALMAR = 1.09      # 관문 ② 2025컷 Calmar ≥ 기준선
GATE_WORST_YEAR = 2.2       # 관문 ③ 최악 해 ≥ +2.2%
GATE_Y2022 = 0.0            # 관문 ④ 2022 > 0

# 기준선 수치(종합표에서 나란히 보여주기 위한 상수 — 재현은 run_v2.py 로 확인됨).
BASELINE_FULL = dict(cagr=13.4, sharpe=1.18, mdd=-12.7, calmar=1.05, worst=2.2, y2022=2.2)
BASELINE_CUT = dict(cagr=10.0, sharpe=1.22, mdd=-9.2, calmar=1.09, worst=2.2)


@dataclass(frozen=True)
class Metrics:
    """한 구성·한 구간의 성과 지표 묶음 + 연도별 수익.

    Attributes:
        cagr / sharpe / mdd / calmar: 표준 성과지표(%·배수).
        worst_year_pct / worst_year: 구간 내 최저 연수익(%)과 그 연도.
        y2022: 2022년 수익(%). 구간에 2022 가 없으면 None.
        yearly: {연도: 수익%} — 연도별 매트릭스 출력용.
        span: 실제 백테스트가 커버한 구간 문자열("YYYY-MM-DD~YYYY-MM-DD").
    """
    cagr: float
    sharpe: float
    mdd: float
    calmar: float
    worst_year_pct: float
    worst_year: int
    y2022: Optional[float]
    yearly: Dict[int, float] = field(default_factory=dict)
    span: str = ""


def metrics_of(res: BacktestResult) -> Metrics:
    """`BacktestResult` 에서 Metrics 를 뽑는다(run_universe37 과 동일 규약)."""
    m = res.metrics["strategy"]
    ys = res.yearly()
    worst = min(ys, key=lambda r: r["strat_pct"])
    y2022 = next((r["strat_pct"] for r in ys if r["year"] == 2022), None)
    yearly = {int(r["year"]): r["strat_pct"] for r in ys}
    span = f"{res.equity.index[0]:%Y-%m-%d}~{res.equity.index[-1]:%Y-%m-%d}"
    return Metrics(cagr=m["cagr_pct"], sharpe=m["sharpe"], mdd=m["mdd_pct"],
                   calmar=m["calmar"], worst_year_pct=worst["strat_pct"],
                   worst_year=int(worst["year"]), y2022=y2022, yearly=yearly, span=span)


def gates(full: Metrics, cut: Metrics) -> List[Tuple[str, bool, str]]:
    """사전 고정 관문 4개를 평가한다(판정 아님 — 산술 통과 여부만).

    Returns:
        [(설명, 통과여부, 실측문자열)] 4개. 어떤 값이 None(구간에 해당 연도 없음)이면
        통과=False 로 두고 실측에 '측정불가' 를 남긴다.
    """
    def ok(v: Optional[float], thr: float) -> Tuple[bool, str]:
        if v is None:
            return False, "측정불가"
        return v >= thr, f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"

    g1 = ok(full.calmar, GATE_FULL_CALMAR)
    g2 = ok(cut.calmar, GATE_CUT_CALMAR)
    g3 = ok(full.worst_year_pct, GATE_WORST_YEAR)
    g4v = (full.y2022 is not None and full.y2022 > GATE_Y2022)
    g4s = "측정불가" if full.y2022 is None else f"{full.y2022:.2f}"
    return [
        (f"① 전체 Calmar ≥ {GATE_FULL_CALMAR}", g1[0], g1[1]),
        (f"② 2025컷 Calmar ≥ {GATE_CUT_CALMAR}", g2[0], g2[1]),
        (f"③ 최악 해 ≥ +{GATE_WORST_YEAR}%", g3[0], f"{full.worst_year_pct:.1f}({full.worst_year})"),
        (f"④ 2022 > 0", g4v, g4s),
    ]


# ── 보고 ────────────────────────────────────────────────────────
def report(title: str, full: Metrics, cut: Metrics,
           years: Optional[List[int]] = None,
           expected_failure: str = "", base_full: Metrics = None,
           base_cut: Metrics = None) -> None:
    """한 실험의 표준 보고: 두 구간 지표표 + 연도별 매트릭스 + 관문 4 + 예상 실패 모드.

    base_full/base_cut 를 주면 같은 창에서 잰 기준선을 나란히 표기한다(창 절단 실험처럼
    기준선 재측정이 필요한 경우). 없으면 동결 상수(BASELINE_*)를 참조로 쓴다.
    """
    logger.info("")
    logger.info(f"════ {title} ════")
    logger.info(f"  구간 실측: 전체 {full.span} · 컷 {cut.span}")

    logger.info(f"  {'구간':<8}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'Calmar':>8}"
                f"{'최저해%':>9}{'2022%':>8}")
    for name, m in (("전체", full), ("2025컷", cut)):
        y22 = f"{m.y2022:>8.2f}" if m.y2022 is not None else f"{'—':>8}"
        logger.info(f"  {name:<8}{m.cagr:>8.1f}{m.sharpe:>8.2f}{m.mdd:>8.1f}"
                    f"{m.calmar:>8.2f}{m.worst_year_pct:>9.1f}{y22}")
    if base_full is not None:
        for name, m in (("기준선전체", base_full), ("기준선컷", base_cut)):
            y22 = f"{m.y2022:>8.2f}" if m.y2022 is not None else f"{'—':>8}"
            logger.info(f"  {name:<8}{m.cagr:>8.1f}{m.sharpe:>8.2f}{m.mdd:>8.1f}"
                        f"{m.calmar:>8.2f}{m.worst_year_pct:>9.1f}{y22}")

    # 연도별 매트릭스
    yrs = years or sorted(full.yearly)
    logger.info(f"  {'연도별%':<8}" + "".join(f"{y:>8}" for y in yrs))
    logger.info(f"  {'실험':<8}" + "".join(
        f"{full.yearly.get(y, float('nan')):>8.1f}" for y in yrs))
    if base_full is not None:
        logger.info(f"  {'기준선':<8}" + "".join(
            f"{base_full.yearly.get(y, float('nan')):>8.1f}" for y in yrs))

    # 관문 4
    logger.info("  관문 4개(통과 여부만 · 판정은 사람):")
    for desc, passed, val in gates(full, cut):
        logger.info(f"      [{'PASS' if passed else 'FAIL'}] {desc:<24} 실측 {val}")
    if expected_failure:
        logger.info(f"  예상 실패 모드: {expected_failure}")


def summary_row(label: str, full: Metrics, cut: Metrics) -> dict:
    """종합표 CSV 한 행."""
    gs = gates(full, cut)
    return {
        "실험": label,
        "전체CAGR%": round(full.cagr, 1), "전체Sharpe": round(full.sharpe, 2),
        "전체MDD%": round(full.mdd, 1), "전체Calmar": round(full.calmar, 3),
        "컷Calmar": round(cut.calmar, 3),
        "최저해%": round(full.worst_year_pct, 1), "최저해": full.worst_year,
        "2022%": round(full.y2022, 2) if full.y2022 is not None else None,
        "관문①": "PASS" if gs[0][1] else "FAIL",
        "관문②": "PASS" if gs[1][1] else "FAIL",
        "관문③": "PASS" if gs[2][1] else "FAIL",
        "관문④": "PASS" if gs[3][1] else "FAIL",
        "관문통과수": sum(1 for g in gs if g[1]),
    }
