"""로더·지표 메모이제이션 — 후보를 수십 개 돌리기 위한 캐시 데코레이터.

워크포워드는 같은 시세·같은 지표를 후보 수만큼 반복해서 읽고 계산한다. 실측(36종 유니버스):
parquet 로드 0.38초 + TrendScore 계산 0.94초 = 후보 1개당 1.3초가 **매번 똑같이** 되풀이된다.
후보 46개면 그것만 60초다. 계산 결과가 후보와 무관하므로(경사 파라미터는 체결 단계에서만
쓰인다) 한 번만 하면 된다.

엔진은 건드리지 않는다 — `DataLoader`/`Indicator` 인터페이스를 그대로 구현한 **위임 래퍼**를
생성자로 주입할 뿐이다(의존성 주입의 이점이 그대로 나오는 자리).

주의 — 두 래퍼는 **짝으로만** 쓴다:
`MemoIndicator` 는 DataFrame 의 객체 식별자(`id`)를 캐시 키로 쓴다. 같은 코드라도 매번 새
DataFrame 이 오면 캐시가 전혀 맞지 않기 때문에, 앞단에 `MemoLoader` 가 있어 **동일 객체가
재사용될 때만** 의미가 있다. 식별자 재활용(객체가 죽고 같은 id 가 다른 객체에 붙는 것)은
캐시가 DataFrame 자체를 함께 들고 있어 원천적으로 막힌다(살아 있는 객체는 id 가 겹치지 않는다).

캐시된 `PriceData` 는 **여러 백테스트가 공유**한다. 엔진은 시세를 읽기만 하므로 안전하지만,
이 로더를 쓰는 코드가 `price.df` 를 제자리 수정하면 다른 실행에 샌다.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

from data import DataLoader, PriceData
from indicator import Indicator

logger = logging.getLogger(__name__)


class MemoLoader(DataLoader):
    """`DataLoader` 위임 래퍼 — 코드별 `PriceData` 를 한 번만 읽는다.

    Args (생성자):
        inner: 실제 로딩을 맡을 로더(예: `ParquetDataLoader`).

    Attributes:
        hits / misses: 캐시 적중·적재 횟수(성능 로그용).
    """

    def __init__(self, inner: DataLoader):
        self.inner = inner
        self._cache: Dict[str, PriceData] = {}
        self.hits = 0
        self.misses = 0

    def load(self, code: str) -> PriceData:
        """코드의 시세를 돌려준다(최초 1회만 실제 로드).

        실패는 캐시하지 않는다 — 실패 사유(파일 없음)가 실행 중에 바뀔 일은 없지만, 예외를
        되던지는 편이 호출부의 fail-loud 가드(`UniverseGuard`)와 같은 모양이 된다.
        """
        cached = self._cache.get(code)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        data = self.inner.load(code)
        self._cache[code] = data
        return data

    def available(self):
        """사용 가능한 종목 코드 목록(위임)."""
        return self.inner.available()

    def log_stats(self) -> None:
        """캐시 효과를 한 줄로 남긴다(디버깅·성능 확인용)."""
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        logger.debug(f"시세 캐시 · 적재 {self.misses}종 · 적중 {self.hits}회 ({rate:.0f}%)")


class MemoIndicator(Indicator):
    """`Indicator` 위임 래퍼 — 같은 DataFrame 에 대한 지표 계산을 한 번만 한다.

    Args (생성자):
        inner: 실제 계산을 맡을 지표(예: `TrendScoreIndicator`).

    Attributes:
        min_len: 위임 지표의 워밍업 봉 수. `IRPBacktesterV2` 의 유니버스 가드가 이 속성을
            읽어 봉 수를 점검하므로 **반드시 그대로 노출**해야 한다(빠지면 점검이 조용히 꺼진다).
    """

    def __init__(self, inner: Indicator):
        self.inner = inner
        # (id(df) → (df, 점수)) — df 를 함께 보관해 캐시가 사는 동안 id 가 재활용되지 않게 한다.
        self._cache: Dict[int, Tuple[pd.DataFrame, pd.Series]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        """표시명(위임)."""
        return self.inner.name

    @property
    def min_len(self):
        """워밍업 봉 수(위임). 없으면 None."""
        return getattr(self.inner, "min_len", None)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """지표 점수를 돌려준다(같은 DataFrame 객체면 캐시)."""
        key = id(df)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is df:
            self.hits += 1
            return cached[1]
        self.misses += 1
        score = self.inner.compute(df)
        self._cache[key] = (df, score)
        return score

    def log_stats(self) -> None:
        """캐시 효과를 한 줄로 남긴다(디버깅·성능 확인용)."""
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        logger.debug(f"지표 캐시 · 계산 {self.misses}회 · 적중 {self.hits}회 ({rate:.0f}%)")
