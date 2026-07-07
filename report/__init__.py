"""리포트 패키지.

백테스트 결과(`BacktestResult`)를 사람이 읽는 산출물로 렌더링한다. 기본 구현은 자체완결형 HTML
(외부 의존 없이 차트를 base64 로 임베드). 새 포맷(PDF, 마크다운 등)은 `Reporter` 하위 클래스로 추가한다.
"""
from .base import Reporter
from .html_report import HTMLReporter

__all__ = ["Reporter", "HTMLReporter"]
