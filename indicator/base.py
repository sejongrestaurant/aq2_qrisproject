"""지표 추상 기반 클래스.

지표는 "표준 스키마 DataFrame → pd.Series" 매핑을 캡슐화하는 값 객체다. 파라미터는 생성자에서
받아 인스턴스가 재사용 가능한 계산기가 되도록 한다(동일 파라미터로 여러 종목에 반복 적용).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    """모든 기술적 지표의 공통 인터페이스.

    하위 클래스는 `compute()` 를 구현한다. 반환 Series 는 입력 DataFrame 인덱스에 정렬되며,
    워밍업 구간은 NaN 으로 둔다(룩어헤드·불완전 계산 방지).
    """

    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """지표 표시명(리포트·컬럼명에 사용)."""
        return self._name

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """표준 스키마 시세로부터 지표 시계열을 계산한다.

        Args:
            data: 소문자 OHLC 컬럼과 DatetimeIndex 를 가진 DataFrame.
        Returns:
            입력 인덱스에 정렬된 지표 값 Series(워밍업 구간 NaN).
        """
        raise NotImplementedError

    # ── 하위 클래스 공용 헬퍼 ────────────────────────────────────────
    @staticmethod
    def _series(x) -> pd.Series:
        """입력을 float Series 로 강제 변환한다."""
        return x if isinstance(x, pd.Series) else pd.Series(x, dtype=float)

    @staticmethod
    def _col(data: pd.DataFrame, *names: str) -> pd.Series | None:
        """후보 이름 중 처음 발견되는 컬럼을 Series 로 반환(없으면 None)."""
        for n in names:
            if n in data.columns:
                return data[n]
        return None

    def __repr__(self) -> str:
        return f"<{self.name}>"
