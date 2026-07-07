"""리포트 추상 기반 클래스.

리포터의 책임은 "백테스트 결과 리스트 → 산출 파일" 하나다. 포맷별 세부는 하위 클래스에 캡슐화한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from backtest import BacktestResult


class Reporter(ABC):
    """모든 리포터의 공통 인터페이스."""

    @abstractmethod
    def generate(self, results: List[BacktestResult], out_path: str, *, title: str = "") -> str:
        """백테스트 결과들을 파일로 렌더링한다.

        Args:
            results: 렌더링할 `BacktestResult` 리스트(1개 이상).
            out_path: 출력 파일 경로.
            title: 리포트 제목(옵션).
        Returns:
            생성된 파일의 절대 경로.
        """
        raise NotImplementedError
