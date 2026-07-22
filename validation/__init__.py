"""검증 계층 — 표본 밖(OOS) 워크포워드로 '동결값이 과최적화인가' 를 묻는다.

`analysis/` 는 이미 나온 자산곡선을 **사후 해석**하는 계층이고, 여기 `validation/` 은 엔진을
**다시 돌려** 판정한다. 나누는 기준은 하나다 — 파라미터 선정을 **학습 창 안에서만** 하게
강제하고 그 선택을 **한 번도 보지 않은 구간**에 적용하는가.

## 왜 필요한가

기준선 수치(전체 CAGR 13.4 · Calmar 1.05)는 2020-01~2026-06 **전 구간을 보고** 고른
파라미터의 성적이다. 그리드 9점이 전부 기준선을 넘겼다(plateau)는 근거가 있어도, 그 그리드
자체를 같은 구간에서 평가했으므로 "표본 밖에서도 그런가" 는 답이 안 된 질문으로 남는다.
발표 Q&A 의 "그거 커브피팅 아닌가요" 에 대한 유일한 정직한 답이 이 계층이다.

## 이 계층은 상품을 바꾸지 않는다

2026-07-15 실험 동결이 발효 중이다. 워크포워드가 어떤 파라미터를 고르든 **동결값
(52/60/0.3)은 그대로다.** 여기서 재선정하는 후보는 '그때 그 시점에 이 규칙을 썼다면 무엇을
골랐을까' 를 재현하기 위한 것이지 채택 후보가 아니다. 산출물은 검증 기록이지 실험 결과가
아니다(규율 위반 아님).

## 구성

  · `memo`        — 로더·지표 메모이제이션(후보 수십 개를 돌리므로 재계산을 캐시).
  · `metrics`     — 자산곡선 조각 → 표준 지표. **재구현하지 않고** `BacktestResult` 에 위임.
  · `windows`     — 학습/검증 창 분할(앵커드=확장 · 롤링=고정폭).
  · `candidates`  — 후보 파라미터 격자 + 후보별 전 구간 곡선 산출·캐시.
  · `selection`   — 학습 창만 보고 후보를 고르는 규칙(규율의 코드화). 최대 Calmar vs plateau.
  · `walkforward` — 오케스트레이터: 창별 선정 → OOS 조각 이어붙이기.
  · `verdict`     — **실행 전에 고정한** 합격 기준과 판정문.
  · `report`      — CSV·차트 산출물.
"""
from __future__ import annotations

from .candidates import Candidate, CandidateGrid, CandidateRunner
from .memo import MemoIndicator, MemoLoader
from .metrics import curve_metrics, slice_growth, yearly_returns
from .selection import MaxMetricRule, PlateauRule, SelectionRule
from .verdict import Verdict, judge
from .walkforward import FoldOutcome, WalkForwardResult, WalkForwardValidator
from .windows import AnchoredWindows, Fold, RollingWindows, WindowScheme

__all__ = [
    "Candidate", "CandidateGrid", "CandidateRunner",
    "MemoIndicator", "MemoLoader",
    "curve_metrics", "slice_growth", "yearly_returns",
    "MaxMetricRule", "PlateauRule", "SelectionRule",
    "Verdict", "judge",
    "FoldOutcome", "WalkForwardResult", "WalkForwardValidator",
    "AnchoredWindows", "Fold", "RollingWindows", "WindowScheme",
]
