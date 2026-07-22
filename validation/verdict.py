"""판정 — **실행 전에 고정한** 합격 기준(규율 1 의 코드화).

규율 1: "판정 기준을 실행 **전에** 고정." 이 파일의 임계값은 워크포워드를 **한 번도 돌리기
전에** 적어 넣은 값이다. 결과를 보고 임계를 옮기면 그 순간 이 검증은 아무것도 검증하지 않는
장식이 된다. 임계를 바꿔야 할 근거가 생기면 **바꾼 사실과 이유를 함께** 기록한다.

임계값 근거:
  · 열화 0.70 — 워크포워드 관행상 OOS/IS 효율 50~70% 를 '살아남았다' 로 본다. 상단을 택했다.
  · 동결 대비 ±0.20 — Calmar 1.0 언저리에서 20%. 동결 그리드 9점의 관측 폭(전체 1.04~1.13)이
    0.09 였으므로, 그 두 배를 '같은 면 위' 로 본다.
  · 벤치 승률 60% — 창이 4~5개뿐이라 3/5 이상을 요구하는 셈. 그 이하는 동전 던지기와 구분 안 됨.
  · 오라클 대비 0.60 — 사후최적의 60% 를 못 따라가면 선정 규칙이 값을 못 한 것으로 본다.
  · 최저 해 ≥ 0 — 상품 서사('잃는 해 없음')를 OOS 에서 그대로 물은 것. 유예 없음.

**이 판정이 답하지 못하는 것(과장 금지):** 검증 구간의 시세도 전부 **과거**다. 여기서
'OOS' 는 '파라미터 선정에 쓰이지 않은 구간' 이라는 뜻이지 미래가 아니다. 창이 4~5개뿐이라
승률의 표준오차가 크고, 유니버스·top_n·히스테리시스·지표 가중은 애초에 전 구간을 보고 정한
값이라 이 검증 밖에 있다(`candidates` 모듈 상단 참조). 따라서 통과는 "동결 선정이 표본 밖에서
재현됐다" 까지만 뒷받침하며, "미래 성과가 보장된다" 는 함의는 없다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .walkforward import (LABEL_BENCH, LABEL_FROZEN, LABEL_GRIDMEAN, LABEL_ORACLE,
                          WalkForwardResult)

logger = logging.getLogger(__name__)

# ── 사전 등록 임계값 (2026-07-22 기록 · 실행 전 고정) ──────────────
MIN_OOS_IS_RATIO = 0.70      # 이어붙인 OOS Calmar ÷ 인샘플(동결 전 구간) Calmar
MAX_FROZEN_GAP = 0.20        # |WF OOS Calmar − 동결 OOS Calmar|
MIN_BENCH_WIN_PCT = 60.0     # 창별 벤치마크 우위 비율(%)
MIN_ORACLE_RATIO = 0.60      # WF OOS Calmar ÷ 오라클 Calmar
MIN_WORST_YEAR_PCT = 0.0     # OOS 최저 해 수익률(%)


@dataclass(frozen=True)
class Check:
    """합격 기준 하나의 판정 결과.

    Attributes:
        name: 기준 이름.
        question: 이 기준이 묻는 질문(사람 말로).
        value / threshold: 실측값과 임계값.
        passed: 통과 여부.
        detail: 값의 단위·비교 방향을 담은 표시 문자열.
    """
    name: str
    question: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True)
class Verdict:
    """워크포워드 판정 결과.

    Attributes:
        rule_name: 판정 대상 선정 규칙.
        checks: 기준별 결과.
        n_passed / n_total: 통과 수 / 전체 수.
        headline: 한 줄 판정문.
    """
    rule_name: str
    checks: List[Check]
    n_passed: int
    n_total: int
    headline: str

    def lines(self) -> List[str]:
        """로그용 판정표(기준별 한 줄 + 결론)."""
        out = [f"[판정 · {self.rule_name}] 사전 등록 기준 {self.n_total}개 중 "
               f"{self.n_passed}개 통과"]
        for c in self.checks:
            mark = "통과" if c.passed else "미달"
            out.append(f"  [{mark}] {c.name:<14} {c.detail}")
            out.append(f"         ↳ {c.question}")
        out.append(f"  ⇒ {self.headline}")
        return out


def judge(result: WalkForwardResult, is_calmar: Optional[float] = None) -> Verdict:
    """사전 등록 기준으로 워크포워드 결과를 판정한다.

    Args:
        result: 판정할 워크포워드 결과.
        is_calmar: 인샘플 기준값 — 동결 파라미터의 **전 구간** Calmar. None 이면 열화 기준을
            건너뛴다(기준 수가 줄어든 사실이 판정문에 그대로 드러난다).
    Returns:
        기준별 통과 여부와 한 줄 판정문을 담은 `Verdict`.
    """
    m = result.oos_metrics
    refs = result.reference_metrics()
    wf_calmar = m["calmar"]
    checks: List[Check] = []

    if is_calmar is not None and is_calmar > 0:
        ratio = wf_calmar / is_calmar
        checks.append(Check(
            name="표본 밖 생존", question="전 구간을 보고 고른 성적이 표본 밖에서도 남는가?",
            value=ratio, threshold=MIN_OOS_IS_RATIO, passed=ratio >= MIN_OOS_IS_RATIO,
            detail=f"OOS/IS Calmar = {wf_calmar:.2f}/{is_calmar:.2f} = {ratio:.2f} "
                   f"(≥{MIN_OOS_IS_RATIO:.2f})"))

    frozen = refs.get(LABEL_FROZEN)
    if frozen is not None:
        gap = abs(wf_calmar - frozen["calmar"])
        checks.append(Check(
            name="동결값 검증", question="동결값이 유난히 운 좋은 한 점이 아니라 평평한 면 위인가?",
            value=gap, threshold=MAX_FROZEN_GAP, passed=gap <= MAX_FROZEN_GAP,
            detail=f"|WF {wf_calmar:.2f} − 동결 {frozen['calmar']:.2f}| = {gap:.2f} "
                   f"(≤{MAX_FROZEN_GAP:.2f})"))

    grid_mean = refs.get(LABEL_GRIDMEAN)
    if grid_mean is not None:
        checks.append(Check(
            name="선정 유용성", question="고르는 것이 아무거나 고르는 것보다 나은가?",
            value=wf_calmar, threshold=grid_mean["calmar"],
            passed=wf_calmar >= grid_mean["calmar"],
            detail=f"WF {wf_calmar:.2f} vs 격자평균 {grid_mean['calmar']:.2f}"))

    oracle = refs.get(LABEL_ORACLE)
    if oracle is not None and oracle["calmar"] > 0:
        ratio = wf_calmar / oracle["calmar"]
        checks.append(Check(
            name="선정 비용", question="사후최적(미리 본 선택) 대비 얼마나 따라가는가?",
            value=ratio, threshold=MIN_ORACLE_RATIO, passed=ratio >= MIN_ORACLE_RATIO,
            detail=f"WF/오라클 = {wf_calmar:.2f}/{oracle['calmar']:.2f} = {ratio:.2f} "
                   f"(≥{MIN_ORACLE_RATIO:.2f})"))

    bench_label = next((lb for lb in refs if lb.startswith(LABEL_BENCH)), None)
    if bench_label is not None:
        win = result.win_rate_vs(bench_label)
        checks.append(Check(
            name="벤치마크 우위", question="창별로 실물 대안(TRF7030)을 이기는가?",
            value=win, threshold=MIN_BENCH_WIN_PCT, passed=win >= MIN_BENCH_WIN_PCT,
            detail=f"창별 승률 {win:.0f}% (≥{MIN_BENCH_WIN_PCT:.0f}%)"))

    worst = m["worst_year_pct"]
    checks.append(Check(
        name="잃는 해 없음", question="'최저 해도 플러스' 라는 서사가 OOS 에서도 성립하는가?",
        value=worst, threshold=MIN_WORST_YEAR_PCT, passed=worst >= MIN_WORST_YEAR_PCT,
        detail=f"OOS 최저 해 {worst:+.1f}% (≥{MIN_WORST_YEAR_PCT:.0f}%)"))

    n_passed = sum(c.passed for c in checks)
    return Verdict(rule_name=result.rule_name, checks=checks, n_passed=n_passed,
                   n_total=len(checks), headline=_headline(n_passed, len(checks)))


def _headline(n_passed: int, n_total: int) -> str:
    """통과 비율에 따른 단계별 판정문(과장하지 않는 표현으로 고정)."""
    if n_total == 0:
        return "판정 불가 — 비교 기준선이 하나도 만들어지지 않았습니다."
    ratio = n_passed / n_total
    if ratio == 1.0:
        return ("사전 등록 기준 전부 통과 — 동결 선정이 표본 밖 구간에서 재현됐다. "
                "(단, 검증 구간도 과거이며 유니버스·top_n·지표 가중은 이 검증 밖에 있다.)")
    if ratio >= 0.66:
        return ("대부분 통과 — 표본 밖 재현을 지지하나 미달 항목의 메커니즘 점검이 필요하다. "
                "미달 항목을 제안서에 그대로 싣고 한계로 서술할 것.")
    if ratio >= 0.34:
        return ("절반 수준 — 표본 밖 재현이 약하다. 동결 수치를 단독으로 제시하지 말고 "
                "OOS 수치를 병기할 것.")
    return ("재현 실패 — 전 구간 성적이 선정 과정에 의존했을 가능성이 크다. "
            "동결 수치를 성과 근거로 쓰지 말 것.")


def compare_rules(verdicts: Dict[str, Verdict]) -> List[str]:
    """규칙별 판정을 나란히 요약한다(선정 규칙 자체의 비교).

    규칙이 달라도 결론이 같으면 "어떻게 골랐든 비슷하다" = plateau 주장의 독립적 재확인이다.
    규칙에 따라 뒤집히면 그 결론은 선정 규칙의 산물이므로 그렇게 서술해야 한다.
    """
    lines = ["[선정 규칙별 판정 요약]", f"  {'규칙':<16}{'통과':>8}  판정"]
    for name, v in verdicts.items():
        lines.append(f"  {name:<16}{v.n_passed:>4}/{v.n_total:<3}  {v.headline.split(' — ')[0]}")
    return lines
