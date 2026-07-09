"""포트폴리오(자산배분) 백테스트 패키지.

단일종목 전략(strategy·backtest)과 별개로, 여러 종목을 목표비중으로 동시 보유하며
리밸런싱하는 자산배분 백테스트를 담는다. 산출물은 기존 `BacktestResult` 로 포장되어
report 계층을 그대로 재사용한다.
"""
from .backtester import PortfolioBacktester
from .config import PortfolioConfig, RebalanceConfig

__all__ = ["PortfolioBacktester", "PortfolioConfig", "RebalanceConfig"]
