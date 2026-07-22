"""학습/검증 창 분할 — 워크포워드의 뼈대.

파라미터를 **학습 창 안에서만** 고르고 그 선택을 **바로 뒤 검증 창**에 적용한 뒤, 창을 앞으로
굴린다. 검증 창들은 겹치지 않으므로 이어붙이면 '한 번도 선정에 쓰이지 않은' 연속 구간이 된다.

두 방식을 둔다:
  · **앵커드(확장)** — 학습 창이 데이터 시작에 고정되고 뒤로만 늘어난다. 실제 운용에 가깝다
    (과거를 버릴 이유가 없다). 다만 뒤 창일수록 학습 표본이 길어져 창끼리 조건이 다르다.
  · **롤링(고정폭)** — 학습 창 길이를 고정해 앞뒤로 민다. 창 조건이 같아 비교가 깔끔하고,
    '오래된 국면을 잊는' 운용을 흉내 낸다. 대신 표본이 짧아 선정이 시끄러워진다.
둘 다 내야 하는 이유: 결론이 방식에 따라 뒤집히면 그 결론은 창 설계의 산물이다.

모든 경계는 **실제 거래일**로 해소한다(달력일로 두면 슬라이스마다 하루씩 어긋난다).
`test_anchor` 는 검증 창 **직전 거래일**이다 — 검증 구간 수익을 재려면 시작 전날 값이
분모로 필요하고, 창을 이어붙일 때 이 겹치는 하루가 이음매가 된다(`metrics.chain`).

워밍업 주의: 학습 창이 데이터 시작보다 뒤에서 열리는 롤링에서도, 엔진은 **전 구간을 한 번에**
돌린 곡선을 자른 것이다. 즉 창 시작 시점의 보유 상태는 그 이전(과거) 데이터로 만들어진
것이지 미래를 본 것이 아니다 — 룩어헤드가 아니라 워밍업이다.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """워크포워드 창 하나(학습 구간 + 그 뒤 검증 구간).

    Attributes:
        index: 창 순번(0부터).
        train_start / train_end: 학습 구간(파라미터 선정에만 쓴다).
        test_anchor: 검증 구간 직전 거래일. 검증 수익의 분모이자 창 이음매.
        test_start / test_end: 검증 구간(선정에 **쓰이지 않은** 구간).
        label: 표시용 라벨(검증 구간 기준). 정확히 한 해면 "2022", 아니면 "2022-01~2022-06".
    """
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_anchor: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str

    @property
    def train_months(self) -> float:
        """학습 구간 길이(개월, 근사). 앵커드에서 창마다 늘어나는 것을 보이기 위한 값."""
        return (self.train_end - self.train_start).days / 30.44

    def describe(self) -> str:
        """로그용 한 줄 설명."""
        return (f"창 {self.index}({self.label}) · 학습 {self.train_start:%Y-%m-%d}~"
                f"{self.train_end:%Y-%m-%d}({self.train_months:.0f}개월) → "
                f"검증 {self.test_start:%Y-%m-%d}~{self.test_end:%Y-%m-%d}")


class WindowScheme(ABC):
    """창 분할 규칙의 공통 인터페이스.

    Args (생성자):
        test_months: 검증 창 길이(개월).
        step_months: 창을 앞으로 미는 간격(개월). test_months 와 같으면 검증 창이 빈틈없이
            이어진다(이어붙이기의 전제) — 다르게 주면 겹치거나 구멍이 나므로 경고한다.
        min_test_months: 마지막 자투리 창을 버리는 하한(개월). 데이터 끝에 두세 달만 남으면
            그 창의 지표는 잡음이라 이어붙이지 않는다.
    """

    name = "window"

    def __init__(self, test_months: int = 12, step_months: Optional[int] = None,
                 min_test_months: int = 6):
        self.test_months = int(test_months)
        self.step_months = int(step_months if step_months is not None else test_months)
        self.min_test_months = int(min_test_months)
        if self.step_months != self.test_months:
            logger.warning(f"[{self.name}] step({self.step_months})≠test({self.test_months}) "
                           f"개월 — 검증 창이 겹치거나 구멍이 생겨 이어붙인 곡선이 왜곡됩니다.")

    # ── public ──────────────────────────────────────────────────────
    def folds(self, index: pd.DatetimeIndex) -> List[Fold]:
        """거래일 인덱스를 받아 창 목록을 만든다.

        Args:
            index: 후보 곡선의 거래일 인덱스(전 구간).
        Returns:
            시간순 `Fold` 리스트.
        Raises:
            ValueError: 데이터가 짧아 창이 하나도 안 나오는 경우(조용히 빈 결과를 주면
                호출부가 '검증했는데 통과' 로 오해한다).
        """
        idx = pd.DatetimeIndex(index)
        # 달 경계로 맞춘다 — 데이터 첫날(1/2 등)에서 개월을 더하면 창이 하루씩 어긋난다.
        first_month = idx[0].normalize().replace(day=1)
        data_end = idx[-1]

        folds: List[Fold] = []
        k = 0
        while True:
            train_end_cal = first_month + pd.DateOffset(months=self._train_span(k)) - pd.Timedelta(days=1)
            if train_end_cal >= data_end:
                break
            test_end_cal = min(train_end_cal + pd.DateOffset(months=self.test_months), data_end)
            # 자투리 창은 버린다. 길이 비교는 **달 단위 오프셋**으로 한다 — 일수를 30.44 로
            # 나눠 재면 딱 6개월인 창(181일)이 5.95 로 계산돼 조용히 잘려 나간다(실제로 2026
            # 상반기 창이 그렇게 사라졌다).
            if test_end_cal < train_end_cal + pd.DateOffset(months=self.min_test_months):
                break
            fold = self._resolve(len(folds), idx, self._train_start_cal(first_month, k),
                                 train_end_cal, test_end_cal)
            if fold is not None:
                folds.append(fold)
            k += 1

        if not folds:
            raise ValueError(
                f"[{self.name}] 창이 하나도 만들어지지 않았습니다 — 데이터 구간"
                f"({idx[0]:%Y-%m-%d}~{data_end:%Y-%m-%d})이 학습 최소 길이 + 검증 "
                f"{self.test_months}개월을 담기에 짧습니다.")
        return folds

    # ── 하위 클래스가 정하는 것 ─────────────────────────────────────
    @abstractmethod
    def _train_span(self, k: int) -> int:
        """k 번째 창에서 '데이터 시작부터 학습 끝까지' 의 개월 수."""
        raise NotImplementedError

    @abstractmethod
    def _train_start_cal(self, first_month: pd.Timestamp, k: int) -> pd.Timestamp:
        """k 번째 창의 학습 시작(달력일)."""
        raise NotImplementedError

    # ── 내부 ────────────────────────────────────────────────────────
    def _resolve(self, i: int, idx: pd.DatetimeIndex, train_start_cal: pd.Timestamp,
                 train_end_cal: pd.Timestamp, test_end_cal: pd.Timestamp) -> Optional[Fold]:
        """달력 경계를 실제 거래일로 해소해 `Fold` 를 만든다(불가하면 None)."""
        train_start = self._first_on_or_after(idx, train_start_cal)
        train_end = self._last_on_or_before(idx, train_end_cal)
        test_end = self._last_on_or_before(idx, test_end_cal)
        if train_start is None or train_end is None or test_end is None:
            return None
        pos = idx.get_loc(train_end)
        if pos + 1 >= len(idx):
            return None
        test_start = idx[pos + 1]
        if test_start > test_end or idx.get_loc(train_end) <= idx.get_loc(train_start):
            return None
        return Fold(index=i, train_start=train_start, train_end=train_end,
                    test_anchor=train_end,          # 검증 직전 거래일 = 학습 마지막 거래일
                    test_start=test_start, test_end=test_end,
                    label=self._label(test_start, test_end))

    @staticmethod
    def _label(test_start: pd.Timestamp, test_end: pd.Timestamp) -> str:
        """검증 구간 라벨. 한 해를 통째로 덮으면 연도만, 아니면 시작~끝 월."""
        if test_start.year == test_end.year and test_start.month == 1 and test_end.month == 12:
            return f"{test_start.year}"
        return f"{test_start:%Y-%m}~{test_end:%Y-%m}"

    @staticmethod
    def _first_on_or_after(idx: pd.DatetimeIndex, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        hit = idx[idx >= day]
        return hit[0] if len(hit) else None

    @staticmethod
    def _last_on_or_before(idx: pd.DatetimeIndex, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        hit = idx[idx <= day]
        return hit[-1] if len(hit) else None


class AnchoredWindows(WindowScheme):
    """확장 학습 창 — 시작점을 데이터 처음에 못 박고 끝만 뒤로 민다.

    Args (생성자):
        min_train_months: 첫 창의 학습 길이(개월). 이 저장소의 지표 워밍업이 252봉(약 12개월)
            이므로 그보다 넉넉해야 선정이 의미를 갖는다. 기본 24개월.
        나머지 인자는 `WindowScheme` 과 동일.
    """

    name = "앵커드(확장)"

    def __init__(self, min_train_months: int = 24, **kwargs):
        super().__init__(**kwargs)
        self.min_train_months = int(min_train_months)

    def _train_span(self, k: int) -> int:
        return self.min_train_months + k * self.step_months

    def _train_start_cal(self, first_month: pd.Timestamp, k: int) -> pd.Timestamp:
        return first_month                                   # 항상 데이터 시작


class RollingWindows(WindowScheme):
    """고정폭 학습 창 — 길이를 유지한 채 앞뒤로 민다.

    Args (생성자):
        train_months: 학습 창 길이(개월). 기본 36개월.
        나머지 인자는 `WindowScheme` 과 동일.
    """

    name = "롤링(고정폭)"

    def __init__(self, train_months: int = 36, **kwargs):
        super().__init__(**kwargs)
        self.train_months = int(train_months)

    def _train_span(self, k: int) -> int:
        return self.train_months + k * self.step_months

    def _train_start_cal(self, first_month: pd.Timestamp, k: int) -> pd.Timestamp:
        return first_month + pd.DateOffset(months=k * self.step_months)
