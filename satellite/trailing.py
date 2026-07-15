"""사테라이트 트레일링 스탑(추적 손절) 규칙 — 다이나믹(ATR) vs 고정(%).

보유 종목이 진입 후 고점 대비 일정 폭 밀리면 청산해 그 슬롯을 현금(BIL)으로 대피시키는
청산 규칙을 정의한다. 계산식만 다른 두 구현체를 **동일 인터페이스**(`stop_level`)로 고정해,
백테스터는 규칙 종류를 몰라도 손절가만 물어보면 되도록 한다(관심사 분리).

두 방식은 헤드투헤드로 비교한다:
  · ATR 다이나믹 — 변동성에 손절폭이 연동(변동성↑ → 폭↑, 휩쏘 완화).
  · 고정 비율    — 변동성과 무관하게 항상 같은 % 후퇴에서 청산(단순·직관).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TrailingStop(ABC):
    """추적 손절선(청산 임계가)을 계산하는 추상 규칙.

    보유구간 고점(`peak`)과 당일 ATR 을 받아, "종가가 이 값 이하로 내려오면 청산" 하는
    손절 수준을 돌려준다. 구현체는 계산식만 맞춘다.

    Attributes:
        name: 리포트·로그 표시명.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    def needs_atr(self) -> bool:
        """ATR 시계열이 필요한 규칙인지(백테스터가 ATR 계산 여부를 결정하는 힌트)."""
        return False

    @abstractmethod
    def stop_level(self, peak: float, atr: float) -> float:
        """보유 고점·당일 ATR 기준 청산 손절가를 반환한다.

        Args:
            peak: 진입 후 지금까지의 종가 고점.
            atr: 당일 ATR(고정 방식은 무시). 워밍업 구간이면 NaN 이 올 수 있고,
                 그 경우 손절가도 NaN 이 되어 트리거되지 않는다(안전).
        """
        raise NotImplementedError


class AtrTrailingStop(TrailingStop):
    """ATR 배수 다이나믹 트레일링: 손절가 = 고점 − mult×ATR(샹들리에식).

    변동성이 커지면(ATR↑) 손절폭이 넓어져 정상 흔들림에 덜 털리고, 잔잔하면 촘촘히 따라붙는다.

    Args (생성자):
        atr_period: ATR 평활 기간(Wilder).
        mult: ATR 배수. 클수록 손절이 느슨(휩쏘↓·손실 확대 가능).
    """

    def __init__(self, atr_period: int = 22, mult: float = 3.0):
        super().__init__(name=f"ATR트레일(×{mult:g})")
        self.atr_period = atr_period
        self.mult = mult

    @property
    def needs_atr(self) -> bool:
        return True

    def stop_level(self, peak: float, atr: float) -> float:
        return peak - self.mult * atr


class FixedTrailingStop(TrailingStop):
    """고정 비율 트레일링: 손절가 = 고점 ×(1 − pct). 변동성과 무관한 일정 % 후퇴 청산.

    Args (생성자):
        pct: 고점 대비 후퇴 비율(0.15 = 고점 −15%에서 청산).
    """

    def __init__(self, pct: float = 0.15):
        super().__init__(name=f"고정트레일({pct * 100:g}%)")
        self.pct = pct

    def stop_level(self, peak: float, atr: float) -> float:
        return peak * (1.0 - self.pct)
