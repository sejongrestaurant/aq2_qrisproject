"""OHLCV 시세 로더.

파이프라인 전체(indicator·strategy·backtest)가 기대하는 **표준 스키마**로 시세를 정규화한다:
  · 컬럼: 소문자 ``open / high / low / close`` (있으면 ``volume`` 도)
  · 인덱스: 오름차순 ``DatetimeIndex``

소스별 세부(parquet, DB, API)는 `DataLoader` 하위 클래스에 캡슐화하고, 상위 계층은
표준 스키마 DataFrame 만 다루게 하여 소스 교체 시 파급을 로더 한 곳으로 국한한다.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

# 표준 스키마 컬럼(대문자→소문자 정규화 매핑 포함)
_OHLC = ["open", "high", "low", "close"]
_RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "Adj Close": "adj_close", "AdjClose": "adj_close",
}


@dataclass
class PriceData:
    """단일 종목의 표준화된 시세 묶음.

    Attributes:
        code: 종목 코드/티커.
        df: 표준 스키마 DataFrame(소문자 OHLC + DatetimeIndex, 오름차순).
        name: 사람이 읽는 표시명(옵션).
    """
    code: str
    df: pd.DataFrame
    name: Optional[str] = None

    def __len__(self) -> int:
        return len(self.df)

    @property
    def start(self) -> pd.Timestamp:
        return self.df.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.df.index[-1]


class DataLoader(ABC):
    """시세 로더 추상 기반 클래스.

    하위 클래스는 `load(code)` 하나만 구현하면 된다. `normalize()` 헬퍼로 어떤 소스든
    표준 스키마로 통일한다.
    """

    @abstractmethod
    def load(self, code: str) -> PriceData:
        """단일 종목 시세를 표준 스키마 `PriceData` 로 로드한다."""
        raise NotImplementedError

    def load_many(self, codes: List[str]) -> Dict[str, PriceData]:
        """여러 종목을 로드해 {code: PriceData} 로 반환. 실패 종목은 조용히 건너뛴다."""
        out: Dict[str, PriceData] = {}
        for code in codes:
            try:
                out[code] = self.load(code)
            except Exception:  # noqa: BLE001 — 개별 종목 실패가 전체를 막지 않도록
                continue
        return out

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """임의 OHLCV DataFrame 을 표준 스키마로 정규화한다.

        · 'Date' 컬럼이 있으면 인덱스로 승격, 없으면 기존 인덱스를 datetime 으로 변환.
        · 대문자 OHLC 컬럼을 소문자로 개명.
        · 오름차순 정렬 + OHLC 결측 행 제거.

        Returns:
            소문자 OHLC(+volume) 컬럼과 오름차순 DatetimeIndex 를 가진 DataFrame.

        Raises:
            ValueError: close 컬럼을 찾지 못한 경우.
        """
        df = df.rename(columns=_RENAME).copy()

        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])
        elif "Date" in df.columns:
            df.index = pd.to_datetime(df["Date"])
            df = df.drop(columns=["Date"])
        else:
            df.index = pd.to_datetime(df.index)

        if "close" not in df.columns:
            raise ValueError("normalize: 'close' 컬럼을 찾을 수 없음 "
                             f"(가용 컬럼: {list(df.columns)})")

        keep = [c for c in _OHLC + ["volume"] if c in df.columns]
        return df[keep].sort_index().dropna(subset=[c for c in _OHLC if c in keep])


class ParquetDataLoader(DataLoader):
    """디렉터리의 ``<code>.parquet`` 파일에서 시세를 읽는 로더.

    한국 ETF(‘Date’ 컬럼)와 미국 ETF(DatetimeIndex)처럼 저장 형태가 달라도
    `normalize()` 가 동일 스키마로 흡수한다.
    """

    def __init__(self, data_dir: str):
        """Args: data_dir: ``<code>.parquet`` 들이 위치한 디렉터리 경로."""
        self.data_dir = data_dir

    def load(self, code: str) -> PriceData:
        path = os.path.join(self.data_dir, f"{code}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"parquet 없음: {path}")
        df = self.normalize(pd.read_parquet(path))
        return PriceData(code=code, df=df)

    def available(self) -> List[str]:
        """디렉터리에 존재하는 종목 코드 목록(파일명 stem)을 반환한다."""
        if not os.path.isdir(self.data_dir):
            return []
        return sorted(
            f[:-len(".parquet")]
            for f in os.listdir(self.data_dir)
            if f.endswith(".parquet")
        )
