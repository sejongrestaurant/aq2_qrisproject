"""기술적 지표 패키지.

모든 지표는 `Indicator` 를 상속하고 표준 스키마 DataFrame(소문자 OHLC + DatetimeIndex)을 입력받아
`pd.Series` 를 반환한다. 새 지표(예: SuperTrend, MACD)는 `Indicator` 하위 클래스로 추가하면
strategy/backtest 계층과 그대로 결합된다.
"""
from .base import Indicator
from .rsi import RSIIndicator
from .adx import ADXIndicator
from .trend_score import TrendScoreIndicator
from .supertrend import SuperTrendIndicator

__all__ = ["Indicator", "RSIIndicator", "ADXIndicator", "TrendScoreIndicator",
           "SuperTrendIndicator"]
