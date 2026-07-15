"""yfinance 온라인 시세 로더.

백테스트 구간(`start`~`end`)을 지정하면, **지표 워밍업 봉 수(`warmup_bars`)만큼 시작일 이전 데이터까지**
추가로 내려받는다. 이렇게 하면 TrendScore(252봉 워밍업)가 백테스트 시작 시점에 이미 예열되어 있어,
사용자가 원한 구간 전체가 유효한 매매 구간이 된다(앞부분이 워밍업으로 버려지지 않음).

선택적 캐시: 내려받은 원본을 `<data_dir>/<code>.parquet` 로 저장/재사용해 반복 실행을 빠르게(오프라인 폴백)한다.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .loader import DataLoader, PriceData

logger = logging.getLogger(__name__)

# 거래일→달력일 환산(연 ~252거래일) + 안전마진. warmup_bars 거래일을 확보하기 위한 달력 버퍼 계수.
_CAL_PER_TRADING = 365.0 / 252.0
_SAFETY = 1.35


class YFinanceDataLoader(DataLoader):
    """yfinance 로 시세를 내려받는 로더(워밍업 자동 확장 + 캐시).

    Args (생성자):
        start: 백테스트 시작일("YYYY-MM-DD" 또는 None=가능한 과거 전체).
        end: 백테스트 종료일("YYYY-MM-DD" 또는 None=최신).
        warmup_bars: 시작일 이전에 추가 확보할 워밍업 거래봉 수(지표 예열용).
        cache_dir: 캐시 parquet 저장 디렉터리(None 이면 캐시 미사용).
        auto_adjust: yfinance 수정주가 사용 여부(기본 False → 원본 OHLC 유지).
    """

    def __init__(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        warmup_bars: int = 260,
        cache_dir: Optional[str] = None,
        auto_adjust: bool = True,
    ):
        self.start = start
        self.end = end
        self.warmup_bars = warmup_bars
        self.cache_dir = cache_dir
        self.auto_adjust = auto_adjust

    # ── public ──────────────────────────────────────────────────────
    def load(self, code: str) -> PriceData:
        """워밍업 포함 구간을 내려받아 표준 스키마 `PriceData` 로 반환한다.

        네트워크 실패 시 캐시(parquet)가 있으면 폴백한다.
        """
        fetch_start = self._fetch_start()
        last_exc = None
        df = None
        for ticker in self._yahoo_candidates(code):
            try:
                df = self._download(ticker, fetch_start, self.end)
                break
            except Exception as exc:  # noqa: BLE001 — 다음 후보(.KS→.KQ)로 폴백
                last_exc = exc
        if df is not None:
            if self.cache_dir:
                self._save_cache(code, df)  # 캐시는 원 코드명으로 저장(파이프라인 호환)
        else:
            cached = self._load_cache(code)
            if cached is None:
                raise RuntimeError(
                    f"{code}: yfinance 다운로드 실패, 캐시도 없음 ({last_exc})") from last_exc
            logger.warning(f"{code} 다운로드 실패 → 캐시 사용 ({last_exc})")
            df = cached
        return PriceData(code=code, df=df)

    @staticmethod
    def _yahoo_candidates(code: str) -> list:
        """KRX 6자리 코드는 .KS → .KQ 순으로 시도, 미국 티커는 그대로."""
        if len(code) == 6 and code.isalnum() and not code.isalpha():
            return [f"{code}.KS", f"{code}.KQ"]
        return [code]

    # ── 내부 ────────────────────────────────────────────────────────
    def _fetch_start(self) -> Optional[str]:
        """백테스트 start 에서 워밍업 봉 수만큼 앞선 다운로드 시작일(달력일)을 계산."""
        if not self.start:
            return None  # 전체 과거를 받으면 워밍업은 자연히 충족
        start_dt = pd.to_datetime(self.start)
        buffer_days = math.ceil(self.warmup_bars * _CAL_PER_TRADING * _SAFETY)
        return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    def _download(self, code: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        """yfinance 다운로드 → MultiIndex 컬럼 평탄화 → 표준 스키마 정규화."""
        import yfinance as yf  # 지연 임포트(로컬 parquet 만 쓸 때 의존 회피)

        raw = yf.download(code, start=start, end=end, progress=False,
                          auto_adjust=self.auto_adjust)
        if raw is None or raw.empty:
            raise ValueError(f"{code}: 빈 응답(구간 {start}~{end})")
        # 단일 종목도 (필드, 티커) 2단 컬럼으로 오므로 티커 레벨 제거
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(-1, axis=1)
        raw = raw.rename_axis(index="date").reset_index()
        return self.normalize(raw)

    # ── 캐시 ────────────────────────────────────────────────────────
    def _cache_path(self, code: str) -> str:
        return os.path.join(self.cache_dir, f"{code}.parquet")

    def _save_cache(self, code: str, df: pd.DataFrame) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        df.to_parquet(self._cache_path(code))

    def _load_cache(self, code: str) -> Optional[pd.DataFrame]:
        if not self.cache_dir:
            return None
        path = self._cache_path(code)
        if not os.path.exists(path):
            return None
        return self.normalize(pd.read_parquet(path))
