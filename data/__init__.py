"""데이터 입출력 패키지.

OHLCV 시세를 표준 스키마(소문자 컬럼 + DatetimeIndex)로 로드하는 로더 클래스를 제공한다.
새 소스(예: DB, REST API)는 `DataLoader` 를 상속해 `load()` 만 구현하면 파이프라인 나머지와 그대로 결합된다.
"""
from .loader import DataLoader, ParquetDataLoader, PriceData
from .yfinance_loader import YFinanceDataLoader

__all__ = ["DataLoader", "ParquetDataLoader", "YFinanceDataLoader", "PriceData"]
