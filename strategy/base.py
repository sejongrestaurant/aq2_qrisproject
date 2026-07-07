"""전략 추상 기반 클래스 + 신호 컨테이너.

전략의 책임은 "표준 스키마 시세 → 봉별 목표 보유상태" 매핑 하나다. 목표 보유상태는 **봉 i 종가에서
확정되는, 익일(i+1) 원하는 롱-플랫 상태** 로 정의한다(백테스트 엔진이 i+1 시가에 체결 → 룩어헤드 방지).
전략은 체결·비용·성과를 알 필요가 없고, 그것은 backtest 계층의 책임이다(관심사 분리).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class Signals:
    """전략이 산출한 신호 묶음.

    Attributes:
        target_long: bool Series. 각 봉에서 "익일 롱 보유를 원하는가". 백테스트가 익일 시가 체결에 사용.
        indicators: 오실레이터형 부가 시계열(0~100 등, 예 {'TrendScore': Series}). 리포트 별도 패널.
        overlays: 가격 수준형 부가 시계열(예 {'SuperTrend': Series}). 리포트에서 가격 차트에 겹쳐 그림.
        둘 다 성과 계산엔 미사용, 시각화·디버깅용.
    """
    target_long: pd.Series
    indicators: Dict[str, pd.Series] = field(default_factory=dict)
    overlays: Dict[str, pd.Series] = field(default_factory=dict)


class Strategy(ABC):
    """모든 매매 전략의 공통 인터페이스.

    하위 클래스는 `generate_signals()` 를 구현한다. 롱-플랫(숏 없음) 스윙을 기본 가정하되,
    확장 시 `Signals` 에 필드를 추가하는 방식으로 목표 비중·숏 등을 도입할 수 있다.
    """

    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """전략 표시명(리포트에 사용)."""
        return self._name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        """표준 스키마 시세로부터 봉별 목표 보유상태를 생성한다.

        Args:
            data: 소문자 OHLC + DatetimeIndex DataFrame.
        Returns:
            `Signals`(target_long bool Series + 부가 지표).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"
