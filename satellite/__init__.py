"""사테라이트(모멘텀 로테이션) 백테스트 패키지.

후보 유니버스에서 지표 점수 상위 top_n 종목을 동일가중 보유하고 체크주기마다 교체하는
전략을 담는다. 산출물은 기존 `BacktestResult` 로 포장되어 report 계층을 그대로 재사용한다.
"""
from .backtester import SatelliteBacktester
from .config import SatelliteConfig, TrailingStopConfig
from .trailing import AtrTrailingStop, FixedTrailingStop, TrailingStop

__all__ = [
    "SatelliteBacktester", "SatelliteConfig", "TrailingStopConfig",
    "TrailingStop", "AtrTrailingStop", "FixedTrailingStop",
]
