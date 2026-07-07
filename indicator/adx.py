"""방향성 ADX 지표 (+ Open-Close / Close-to-Close DM 변형).

표준 ADX(추세 강도, 0~100)에 **부호(방향)** 를 곱해 반환한다: +DI 우세면 양수, -DI 우세면 음수,
방향이 불분명(잡음 임계 이내)하면 0. 참고 트리 `indicator_util.calculate_adx_series` 와 동일 로직으로,
TrendScore 의 ADX soft-penalty(추세 강도가 약하면 점수 차감) 계산에 쓰인다.

표준 DMI 는 방향성 이동(±DM)을 **고가/저가 극단**으로만 계산(+DM=High−prevHigh, −DM=prevLow−Low)해서,
장중 고저는 거의 안 변하는데 종가만 계속 오르는 추세(1%씩 매일 상승 등)에서 DM 이 작아 **ADX 가 늦게 반응**한다.
이를 보완하려고 종가/시가 기반 DM 을 섞는 변형을 ``dm_mode`` 로 선택할 수 있다:

  · ``highlow``    (기본) 표준 DMI. 하위호환.
  · ``close``      종가-대-종가 DM(±=max(±close.diff(),0)). 종가 모멘텀의 ADX화, 노이즈 적음(가장 흔함).
  · ``body``       캔들 몸통 DM(±=max(±(close−open),0)). 양봉/음봉 방향을 직접 반영, 반응 빠름.
  · ``hybrid_avg`` 0.5·highlow + 0.5·close. 장중 고저 정보 유지 + 종가 가속 반영.
  · ``hybrid_max`` max(highlow, close) 사이드별. 둘 중 강한 신호 채택.

또 True Range 에 몸통(|close−open|)을 섞어 분모를 조정할 수 있다: ``tr_body_alpha`` (1.0=순수 TR, 0.0=순수 몸통).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator

_DM_MODES = ("highlow", "close", "body", "hybrid_avg", "hybrid_max")


class ADXIndicator(Indicator):
    """부호를 가진 방향성 ADX (DM 계산 방식 선택 가능).

    Args (생성자):
        period: 평활 기간(기본 14).
        dm_mode: 방향성 이동 계산 방식(highlow|close|body|hybrid_avg|hybrid_max). 위 모듈 설명 참조.
        tr_body_alpha: True Range 몸통 혼합비. TR=α·TrueRange+(1−α)·|close−open|. 1.0=순수 TR(기본).
    """

    def __init__(self, period: int = 14, dm_mode: str = "highlow",
                 tr_body_alpha: float = 1.0, name: str | None = None):
        if dm_mode not in _DM_MODES:
            raise ValueError(f"dm_mode 는 {_DM_MODES} 중 하나여야 함: {dm_mode}")
        if not 0.0 <= tr_body_alpha <= 1.0:
            raise ValueError(f"tr_body_alpha 는 0~1 이어야 함: {tr_body_alpha}")
        suffix = "" if dm_mode == "highlow" else f"·{dm_mode}"
        super().__init__(name or f"ADX{period}{suffix}")
        self.period = period
        self.dm_mode = dm_mode
        self.tr_body_alpha = tr_body_alpha

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high = self._col(data, "high", "High")
        low = self._col(data, "low", "Low")
        close = self._col(data, "close", "Close", "adj_close")
        open_ = self._col(data, "open", "Open")
        if close is None:
            raise ValueError("ADXIndicator: 'close' 컬럼 필요")
        # high/low 결측 시 close 로 대체(저하되지만 동작). open 결측이면 body/tr 혼합은 close 로 폴백.
        high = close if high is None else high
        low = close if low is None else low
        return self.from_ohlc(
            self._series(high), self._series(low), self._series(close), self.period,
            open_=None if open_ is None else self._series(open_),
            dm_mode=self.dm_mode, tr_body_alpha=self.tr_body_alpha)

    @staticmethod
    def from_ohlc(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14, open_: pd.Series | None = None,
                  dm_mode: str = "highlow", tr_body_alpha: float = 1.0) -> pd.Series:
        """OHLC 로부터 부호 있는 ADX 를 계산(다른 지표에서 재사용).

        dm_mode/tr_body_alpha 로 방향성 이동·변동폭 계산 방식을 선택한다(모듈 설명 참조).
        open_ 이 None 이면 body/몸통 혼합은 종가 기준으로 자동 폴백한다.
        """
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        if tr_body_alpha < 1.0:                       # TR 에 몸통 혼합(반응 가속)
            body_abs = (close - open_).abs() if open_ is not None else close.diff().abs()
            tr = tr_body_alpha * tr + (1.0 - tr_body_alpha) * body_abs

        # 후보 1: 고저 극단 DM(표준). 봉당 한쪽만 양수.
        up_move, down_move = high.diff(), -low.diff()
        hl_plus = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
        hl_minus = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

        # 후보 2: 종가-대-종가 DM(모멘텀). 후보 3: 캔들 몸통 DM(open 없으면 종가로 폴백).
        chg = close.diff()
        cl_plus, cl_minus = chg.clip(lower=0), (-chg).clip(lower=0)
        body = (close - open_) if open_ is not None else chg
        bd_plus, bd_minus = body.clip(lower=0), (-body).clip(lower=0)

        if dm_mode == "highlow":
            plus_dm, minus_dm = hl_plus, hl_minus
        elif dm_mode == "close":
            plus_dm, minus_dm = cl_plus, cl_minus
        elif dm_mode == "body":
            plus_dm, minus_dm = bd_plus, bd_minus
        elif dm_mode == "hybrid_avg":
            plus_dm, minus_dm = 0.5 * hl_plus + 0.5 * cl_plus, 0.5 * hl_minus + 0.5 * cl_minus
        else:  # hybrid_max
            plus_dm = pd.concat([hl_plus, cl_plus], axis=1).max(axis=1)
            minus_dm = pd.concat([hl_minus, cl_minus], axis=1).max(axis=1)

        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        adx = dx.ewm(com=period - 1, min_periods=period).mean()

        # 방향 판정: DI 격차가 잡음 임계(동적)를 넘을 때만 부호 부여
        di_diff = plus_di - minus_di
        threshold = np.maximum(
            3.0, di_diff.rolling(window=50, min_periods=period).std().fillna(0) * 0.5)
        direction = np.where(di_diff > threshold, 1, np.where(di_diff < -threshold, -1, 0))
        return adx * pd.Series(direction, index=adx.index)
