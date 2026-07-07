"""백테스트 패키지.

전략이 만든 목표 보유상태(롱-플랫)를 받아 익일 시가 체결·거래비용을 적용해 자산곡선과 성과지표를
계산한다. 엔진(`Backtester`)·거래기록(`Trade`)·결과(`BacktestResult`)로 책임을 분리했다.
"""
from .trade import Trade
from .result import BacktestResult
from .engine import Backtester

__all__ = ["Trade", "BacktestResult", "Backtester"]
