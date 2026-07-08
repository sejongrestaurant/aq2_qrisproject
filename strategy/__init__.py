"""매매 전략 패키지.

전략은 표준 스키마 시세를 입력받아 **봉별 목표 보유상태(롱-플랫 bool Series)** 를 생성한다.
백테스트 엔진이 이 신호를 받아 익일 시가 체결로 성과를 계산한다. 새 전략은 `Strategy` 하위 클래스로
추가하면 backtest/report 계층과 그대로 결합된다.
"""
from .base import Strategy, Signals
from .swing_trend_score import TrendScoreSwingStrategy
from .swing_supertrend import SuperTrendSwingStrategy
from .swing_regime_trend_score import RegimeGatedTrendScoreStrategy
from .swing_sma_slope import SMASlopeROCStrategy
from .swing_regime_trendrider import RegimeTrendRiderStrategy

__all__ = ["Strategy", "Signals", "TrendScoreSwingStrategy", "SuperTrendSwingStrategy",
           "RegimeGatedTrendScoreStrategy", "SMASlopeROCStrategy", "RegimeTrendRiderStrategy"]
