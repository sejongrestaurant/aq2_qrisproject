"""납입 계획 — 투자자가 언제 얼마를 넣는지를 현금흐름으로 만든다.

IRP 는 목돈을 한 번에 넣는 상품이 아니라 **매달 월급에서 떼어 붓는** 상품이다. 그래서
'2020년에 1,000만 원을 넣었다면' 식의 일시납 백테스트는 실제 가입자 경험과 다르다.
이 모듈은 두 가지 납입 방식을 같은 인터페이스로 만들어, 같은 자산곡선 위에서 비교한다.

납입일은 **거래일 축에 스냅**한다(달력 1일이 휴장이면 그달 첫 거래일). 주기 판정은
`portfolio.schedule.period_mask` 를 그대로 재사용한다 — 리밸런싱 주기와 같은 규약을 쓰면
'월초' 의 정의가 계층마다 갈리지 않는다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from portfolio.schedule import period_mask

# IRP 세액공제 한도(연 900만 원)를 채우는 납입액. 월 75만 × 12 = 900만.
DEFAULT_MONTHLY: float = 750_000.0
DEFAULT_ANNUAL: float = 9_000_000.0


def _contribution_days(index: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """납입일(각 주기의 첫 거래일 + **가입 첫날**)을 고른다.

    `period_mask` 는 주기가 '바뀌는' 첫 거래일만 True 로 두므로 인덱스 0 은 절대 True 가
    되지 않는다(리밸런싱에선 옳다 — 첫날은 이미 목표비중이니 되돌릴 게 없다). 하지만 납입은
    가입 첫날에 일어나므로 그대로 쓰면 첫 회차가 통째로 빠진다. 롤링 분석에선 창마다 첫
    납입이 사라져 결과가 조용히 뒤틀린다. 그래서 여기서 첫날을 명시적으로 켠다.
    """
    if len(index) == 0:
        return index[:0]
    mask = period_mask(index, period)
    mask[0] = True
    return index[mask]


class CashflowPlan(ABC):
    """납입 계획 인터페이스.

    Attributes:
        name: 표시명(리포트 라벨).
    """

    name: str

    @abstractmethod
    def generate(self, index: pd.DatetimeIndex) -> pd.Series:
        """거래일 축 위에 납입 현금흐름(날짜 → 납입액)을 만든다.

        Args:
            index: 투자 구간의 거래일 인덱스(자산곡선의 인덱스).
        Returns:
            납입이 일어나는 거래일만 담은 Series(양수 = 납입액). 빈 Series 일 수 있다.
        """
        raise NotImplementedError


class MonthlyDCA(CashflowPlan):
    """월 적립식 — 매월 첫 거래일에 같은 금액을 납입한다(월급 투자 패턴).

    Args:
        amount: 월 납입액(기본 75만 원 = 연 900만 세액공제 한도).
    """

    def __init__(self, amount: float = DEFAULT_MONTHLY):
        self.amount = float(amount)
        self.name = f"월 적립 {amount / 10_000:.0f}만"

    def generate(self, index: pd.DatetimeIndex) -> pd.Series:
        days = _contribution_days(index, "M")
        return pd.Series(self.amount, index=days, name="contribution")


class AnnualLump(CashflowPlan):
    """연초 일시납 — 매년 첫 거래일에 그 해 한도를 한 번에 납입한다.

    적립식과 **연 납입액은 같지만 타이밍이 다르다**(연초에 몰아넣음). 상승장에선 일찍 넣은
    쪽이 유리하고 하락장에선 불리하다 — 그 차이를 재는 것이 이 계획의 목적이다.

    Args:
        amount: 연 납입액(기본 900만 원).
    """

    def __init__(self, amount: float = DEFAULT_ANNUAL):
        self.amount = float(amount)
        self.name = f"연초 일시납 {amount / 10_000:.0f}만"

    def generate(self, index: pd.DatetimeIndex) -> pd.Series:
        days = _contribution_days(index, "Y")
        return pd.Series(self.amount, index=days, name="contribution")
