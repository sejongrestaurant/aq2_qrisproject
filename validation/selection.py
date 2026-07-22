"""선정 규칙 — 학습 창만 보고 후보 하나를 고른다(실험 규율의 코드화).

규율 1 은 "판정 기준을 실행 **전에** 고정" 하라고 말하고, 규율 3 은 "채택하려면 그리드에서
plateau 여야 한다(한 점만 좋으면 기각)" 고 말한다. 사람이 지키던 이 두 문장을 함수로 바꾼
것이 이 모듈이다. 규칙을 객체로 만들면 **규칙 자체를 OOS 에서 비교**할 수 있다 —
"최대 Calmar 로 고르는 것과 plateau 로 고르는 것 중 무엇이 표본 밖에서 더 버티나" 는
이 프로젝트가 실제로 답을 가진 적 없는 질문이다.

규칙은 **검증 구간을 절대 보지 않는다.** `select()` 에 넘어오는 점수는 학습 창 지표뿐이고,
검증 창 지표는 `walkforward` 가 선정이 끝난 **뒤에** 계산한다.

동점 처리는 결정적이어야 한다(같은 입력 → 항상 같은 선택). 부동소수 동점에서 dict 순서에
기대면 재현이 깨지므로, 순위 키를 명시적으로 쌓는다: 주지표 → 최저 해 → 낙폭 → 라벨.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .candidates import Candidate, CandidateGrid

logger = logging.getLogger(__name__)


class SelectionRule(ABC):
    """학습 창 지표 → 후보 하나를 고르는 규칙.

    Args (생성자):
        metric: 주 판정 지표 키(`metrics.curve_metrics` 의 키). 기본 "calmar" —
            규율 1 이 "CAGR 단독 비교 금지" 를 못 박았고, 실제 판정에 쓴 지표가 Calmar 다.
    """

    name = "규칙"

    def __init__(self, metric: str = "calmar"):
        self.metric = metric

    @abstractmethod
    def score(self, label: str, train: Dict[str, Dict[str, float]],
              grid: CandidateGrid) -> float:
        """후보 하나의 선정 점수(클수록 좋다).

        Args:
            label: 평가할 후보 라벨.
            train: 라벨 → 학습 창 지표 dict(모든 후보분).
            grid: 이웃 조회용 격자.
        """
        raise NotImplementedError

    # ── public ──────────────────────────────────────────────────────
    def select(self, train: Dict[str, Dict[str, float]], grid: CandidateGrid) -> Candidate:
        """학습 창 지표만 보고 후보를 고른다(동점은 결정적으로 해소).

        Returns:
            선정된 `Candidate`.
        Raises:
            ValueError: 평가할 후보가 없는 경우.
        """
        if not train:
            raise ValueError(f"[{self.name}] 학습 창 지표가 비어 후보를 고를 수 없습니다.")
        ranked = sorted(train.keys(), key=lambda lb: self._sort_key(lb, train, grid),
                        reverse=True)
        return grid.get(ranked[0])

    def ranking(self, train: Dict[str, Dict[str, float]], grid: CandidateGrid,
                top: int = 3) -> List[Tuple[str, float]]:
        """선정 점수 상위 후보(라벨, 점수) — 선정이 얼마나 아슬아슬했는지 로그로 보이기 위한 것."""
        ranked = sorted(train.keys(), key=lambda lb: self._sort_key(lb, train, grid),
                        reverse=True)
        return [(lb, self.score(lb, train, grid)) for lb in ranked[:top]]

    # ── 내부 ────────────────────────────────────────────────────────
    def _sort_key(self, label: str, train: Dict[str, Dict[str, float]],
                  grid: CandidateGrid) -> Tuple[float, float, float, str]:
        """정렬 키: (선정점수, 최저 해, -|MDD|, 라벨 역순).

        2·3순위는 규율 1 의 나머지 기준(최저 해 개선 · 낙폭 방어)을 동점 처리에 반영한 것이다.
        마지막 라벨 키는 재현성 전용(의미 없는 tie 를 알파벳으로 고정).
        """
        m = train[label]
        return (self.score(label, train, grid),
                float(m.get("worst_year_pct", float("-inf"))),
                -abs(float(m.get("mdd_pct", 0.0))),
                label)


class MaxMetricRule(SelectionRule):
    """가장 단순한 규칙 — 학습 창 지표가 가장 높은 후보를 고른다.

    과최적화에 그대로 노출되는 순진한 규칙이라, plateau 규칙의 **대조군**으로 필요하다.
    이 규칙이 OOS 에서 크게 무너지고 plateau 는 버틴다면, 규율 3 이 값을 했다는 증거가 된다.
    """

    def __init__(self, metric: str = "calmar"):
        super().__init__(metric)
        self.name = f"최대 {metric}"

    def score(self, label: str, train: Dict[str, Dict[str, float]],
              grid: CandidateGrid) -> float:
        """학습 창 주지표 그대로."""
        return float(train[label].get(self.metric, float("-inf")))


class PlateauRule(SelectionRule):
    """규율 3 의 코드화 — **자기 + 이웃 평균**이 가장 높은 후보를 고른다.

    한 점만 뾰족하게 좋은 후보는 이웃이 낮아 평균이 깎이고, 면 전체가 들린 구역의 한가운데
    후보가 이긴다. 즉 "주변값에서도 성립하는가" 를 선정 단계에서 강제한다.

    Args (생성자):
        metric: 주 판정 지표 키.
        neighbor_weight: 이웃 점수의 가중(0 이면 `MaxMetricRule` 과 같아진다). 1.0 이면
            자기와 이웃을 동등하게 평균한다. 기본 1.0 — 임의의 중간값을 두면 그 자체가
            튜닝할 손잡이가 되어버려, 해석 가능한 양 끝 중 하나를 택했다.
        require_neighbors: True 면 이웃이 없는 후보(격자 밖, 예 V1)를 **제외**한다.
            기본 False — V1 은 귀무가설이라 후보로 남겨 두고, 이웃이 없으니 자기 점수로만
            겨루게 한다(격자 후보는 이웃 평균이라 대개 불리해지지 않는다).
    """

    def __init__(self, metric: str = "calmar", neighbor_weight: float = 1.0,
                 require_neighbors: bool = False):
        super().__init__(metric)
        self.neighbor_weight = float(neighbor_weight)
        self.require_neighbors = bool(require_neighbors)
        self.name = f"plateau {metric}"

    def score(self, label: str, train: Dict[str, Dict[str, float]],
              grid: CandidateGrid) -> float:
        """자기 점수와 이웃 점수의 가중 평균.

        이웃 중 학습 지표가 없는 것(격자에는 있으나 이번 실행에서 안 돈 후보)은 건너뛴다.
        """
        own = float(train[label].get(self.metric, float("-inf")))
        nb = [float(train[n.label][self.metric])
              for n in grid.neighbors(grid.get(label)) if n.label in train]
        if not nb:
            return float("-inf") if self.require_neighbors else own
        return (own + self.neighbor_weight * (sum(nb) / len(nb))) / (1.0 + self.neighbor_weight)


def default_rules() -> List[SelectionRule]:
    """이 프로젝트에서 함께 돌릴 기본 규칙 세트.

    셋을 나란히 내는 이유: 결론이 규칙에 따라 뒤집히는지 보기 위해서다. 뒤집히지 않으면
    "어떻게 골랐든 비슷하다" = 면이 평평하다는 plateau 주장의 **독립적인 재확인**이 된다.
    """
    return [PlateauRule("calmar"), MaxMetricRule("calmar"), MaxMetricRule("sharpe")]
