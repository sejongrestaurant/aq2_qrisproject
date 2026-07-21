"""[C-2] 국면 연동 채권비중 — 하락장 판정 시 코어(채권) 30→50% 로 확대.

top-down 국면 스위치다(전달문: 철학과 다르지만 측정은 한다). KOSPI200(069500) 종가가
200일 이동평균 아래면 '하락장' 으로 보고, 그 구간 리밸런싱에서 채권 슬리브 목표비중을
30%→bear_bond_weight(기본 50%)로 올리고 사테라이트를 그만큼 줄인다.

원본 `_simulate` 은 목표비중 `w_t` 가 고정이다. 이 변형은 리밸런싱 시점마다 국면에 따라 두
목표비중(상승장/하락장) 중 하나를 고른다. 그 외(수익 반영·회전율 비용·임계 트리거·룩어헤드)는
원본과 동일하다 — `_simulate` 만 시그니처를 유지한 채 오버라이드해 `super().run()` 이 다형적으로
이 버전을 부르게 한다(나머지 조립·벤치마크·결과 포장은 원본 재사용).

국면 판정은 종가 기준이라 룩어헤드가 없다(당일 200MA 로 당일 리밸런싱 목표를 정한다 —
결정과 실행이 같은 날 종가). 예상 관전점: 2022 하락장에서 채권비중이 실제로 올라가 방어가
강화되는가, 아니면 2020·2023·2025 V자 반등 초입에 채권을 늘려 재진입을 더 늦추는가.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from irp.backtester_v2 import IRPBacktesterV2
from portfolio.schedule import period_mask

logger = logging.getLogger(__name__)

_SAT_KEY = "__SAT__"


class RegimeBondBacktester(IRPBacktesterV2):
    """하락장에서 채권비중을 키우는 IRP 백테스터.

    Args (생성자):
        bear_bond_weight: 하락장 채권 슬리브 총비중(기본 0.5). 상승장은 설정값(0.3) 유지.
        regime_ticker: 국면 판정 지수 대용(기본 069500 KODEX200).
        ma_window: 이동평균 창(기본 200거래일).
        나머지: `IRPBacktesterV2` 와 동일.
    """

    def __init__(self, loader, indicator, cost: float = 0.0010, *,
                 satellite=None, allow_missing: bool = False,
                 bear_bond_weight: float = 0.5,
                 regime_ticker: str = "069500", ma_window: int = 200):
        super().__init__(loader=loader, indicator=indicator, cost=cost,
                         satellite=satellite, allow_missing=allow_missing)
        self.bear_bond_weight = float(bear_bond_weight)
        self.regime_ticker = regime_ticker
        self.ma_window = int(ma_window)
        self._bear_frac: Optional[float] = None  # 진단용: 하락장 판정 비율

    # ── 국면 마스크 ─────────────────────────────────────────────────
    def _bear_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """index 각 날짜의 하락장 여부(종가 < 200MA)를 bool 배열로.

        200MA 는 지수 **전체 이력**으로 계산한 뒤 백테스트 날짜축에 정렬한다(구간 시작에서
        이미 예열된 200MA 를 쓰기 위해 — 그렇지 않으면 초기 200일이 판정 불가가 된다)."""
        close = self.loader.load(self.regime_ticker).df["close"]
        ma = close.rolling(self.ma_window).mean()
        bear = (close < ma).reindex(index).ffill().fillna(False)
        return bear.to_numpy(dtype=bool)

    def _weight_vectors(self, cols: List[str], weights) -> Tuple[np.ndarray, np.ndarray]:
        """(상승장, 하락장) 목표비중 벡터. 하락장은 채권 총비중을 bear_bond_weight 로 올린다."""
        w_bull = np.array([weights[c] for c in cols], dtype=float)
        sat_i = cols.index(_SAT_KEY)
        bull_bond_total = 1.0 - w_bull[sat_i]
        scale = self.bear_bond_weight / bull_bond_total  # 채권 각각을 같은 비율로 확대
        w_bear = w_bull * scale
        w_bear[sat_i] = 1.0 - self.bear_bond_weight
        return w_bull, w_bear

    # ── 시뮬레이션(국면 연동 목표비중) ──────────────────────────────
    def _simulate(self, rets: pd.DataFrame, weights, period: str,
                  threshold: Optional[float] = None
                  ) -> Tuple[pd.Series, List[pd.Timestamp]]:
        """원본 `_simulate` 과 동일하되, 리밸런싱 목표비중만 국면에 따라 고른다."""
        cols = list(rets.columns)
        w_bull, w_bear = self._weight_vectors(cols, weights)
        sat_i = cols.index(_SAT_KEY)
        R = rets.to_numpy()
        dates = rets.index
        n = len(dates)
        periodic = period_mask(dates, period)
        bear = self._bear_mask(dates)
        self._bear_frac = float(bear.mean())

        value = w_bull.copy()  # 시작은 상승장 가정(첫날 즉시 리밸런싱되지 않음)
        eq = np.empty(n)
        rb_dates: List[pd.Timestamp] = []
        for i in range(n):
            if i > 0:
                value = value * (1.0 + R[i])
            total = value.sum()
            w_now = value / total
            w_t = w_bear if bear[i] else w_bull       # ★ 국면 연동 목표비중
            # 임계 트리거는 '현 목표비중 대비' 사테라이트 이탈로 판정(국면이 바뀌면 목표도 바뀜).
            drift_hit = (threshold is not None
                         and abs(w_now[sat_i] - w_t[sat_i]) > threshold)
            if i > 0 and (periodic[i] or drift_hit):
                turnover = 0.5 * np.abs(w_now - w_t).sum()
                total *= (1.0 - self.cost * turnover)
                value = w_t * total
                rb_dates.append(dates[i])
            eq[i] = total
        logger.info(f"국면연동 채권비중 · 하락장 판정 {self._bear_frac * 100:.0f}% 구간 · "
                    f"상승장 채권 {(1 - w_bull[sat_i]) * 100:.0f}% / 하락장 "
                    f"{self.bear_bond_weight * 100:.0f}% · 리밸런싱 {len(rb_dates)}회")
        return pd.Series(eq, index=dates, name="equity"), rb_dates
