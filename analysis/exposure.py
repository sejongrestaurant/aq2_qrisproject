"""실효 노출률 측정 — 동결 V2 슬리브가 실제로 얼마나 투자돼 있었나.

**왜 이 모듈이 필요한가.** 구 지표였던 "슬롯 미달 개월 수"는 V1(이진 게이트)의 언어다.
V1 은 슬롯을 채우거나(1/top_n) 안 채우거나 둘 뿐이라 '미달 개월'이 잘 정의된다. 그러나 동결
V2 에는 이진 게이트가 없다 — 52 점 30% → 60 점 100% 의 **부분 충전**이므로 "슬롯이 찼다"가
30% 충전을 포함해 버린다. 같은 '만충 7슬롯'이 실제로는 70% 노출일 수도, 30% 노출일 수도 있다.
이진 개월 수는 V2 에서 정의가 어긋나고 노출을 과대평가한다.

그래서 공식 지표를 **실효 노출률**(만충 대비 실제 투자 비율)로 둔다. 슬롯 수는 국면 대응이라는
원 설계를 그대로 계승하되(위기에 셔터가 내려간다), V2 의 정체성인 '계단 → 경사로'가 수치에
드러난다.

**측정 방식 — 재계산하지 않고 관측한다.** 점수·히스테리시스·자격 판정을 이 모듈이 다시 구현하면
엔진과 미세하게 어긋나 결과를 오염시킨다(실제로 구 `universe_availability.py` 가 그랬다: 월말
resample 로 쟀으나 엔진의 체크 시점은 월초 첫 거래일이었고, 창을 안 잘라 캐시가 하루 더 긴
종목 하나 때문에 존재하지 않는 달을 만들어 냈다). 이 프로브는 엔진이 **이미 정한** 목표비중을
받아 적기만 하므로, 기록값은 시뮬레이션이 실제로 쓴 값과 정의상 일치한다. 측정 창도 엔진의
날짜축 그 자체라 창 불일치가 구조적으로 불가능하다.

`ramp_score=None` 이면 상위 클래스가 원본 이진 게이트로 축퇴하므로(부동소수점까지 동일),
**같은 프로브로 V1(계단)과 V2(경사로)를 같은 코드 경로에서** 잰다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.schedule import period_mask
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Check:
    """체크 시점 한 번의 관측치(내부용)."""
    fill: float          # 슬리브 충전율 0~1 (= 1 − 현금비중)
    slots_used: int      # 자격을 얻어 소비된 슬롯 수(충전율 무관)
    n_valid: int         # 점수·가격이 모두 있는 후보 수(워밍업·상장 완료)
    n_gate: int          # 그중 신규 자격 문턱을 넘은 수


@dataclass(frozen=True)
class ExposureResult:
    """체크 시점별 사테라이트 노출과 그 원인.

    Attributes:
        fill: 슬리브 충전율(0~1). 1.0 = 슬롯 top_n 개 만충.
        slots_used: 소비된 슬롯 수. **충전율과 무관하다** — 30%만 채우는 종목도 슬롯 하나를
            차지한다(그래서 slots_used 는 노출의 상한만 말해 준다).
        n_valid: 유효 후보 수(데이터 공백 실태).
        n_gate: 자격 문턱 통과 수(게이트 실태).
        sleeve_weight: 포트폴리오에서 사테라이트가 갖는 비중(예 0.70).
        top_n: 슬롯 수.
    """
    fill: pd.Series
    slots_used: pd.Series
    n_valid: pd.Series
    n_gate: pd.Series
    sleeve_weight: float
    top_n: int

    @property
    def portfolio_exposure(self) -> pd.Series:
        """포트폴리오 기준 위험자산 노출(= 충전율 × 슬리브 비중). 만충이면 sleeve_weight."""
        return (self.fill * self.sleeve_weight).rename("portfolio_exposure")

    @property
    def shortfall_cause(self) -> pd.Series:
        """만충이 아닌 체크의 원인 — 데이터 공백인가 게이트인가.

        이 분해가 제안서의 논지다: 노출이 낮은 것은 **의도된 방어**(게이트)이지 데이터 결함이
        아니다. 유효 후보가 top_n 보다 적으면 데이터 공백 탓, 후보는 충분한데 자격 미달이면
        게이트 탓이다.
        """
        cause = pd.Series("만충", index=self.fill.index, name="shortfall_cause")
        short = self.slots_used < self.top_n
        data_bound = self.n_valid < self.top_n
        cause[short & data_bound] = "데이터 공백(유효 후보 < top_n)"
        cause[short & ~data_bound] = "게이트 미달(후보는 충분)"
        # 슬롯은 다 찼는데 부분 충전이라 노출이 만충이 아닌 경우 — V2 에만 있는 상태.
        cause[~short & (self.fill < 1.0 - 1e-9)] = "부분 충전(슬롯은 만충)"
        return cause

    def to_frame(self) -> pd.DataFrame:
        """월별 상세표(CSV 용)."""
        return pd.DataFrame({
            "충전율%": (self.fill * 100).round(2),
            "포트폴리오노출%": (self.portfolio_exposure * 100).round(2),
            "소비슬롯": self.slots_used,
            "유효후보": self.n_valid,
            "자격통과": self.n_gate,
            "상태": self.shortfall_cause,
        })


class ExposureProbe(SatelliteBacktesterV2):
    """슬롯 충전율과 그 원인을 체크 시점마다 기록하는 관측용 사테라이트 백테스터.

    엔진 로직은 일절 바꾸지 않는다 — 상위 클래스의 `_targets()` 를 그대로 호출하고 결과만
    받아 적는다. 따라서 이 프로브를 끼운 백테스트의 자산곡선은 끼우지 않은 것과 **동일**하다
    (러너가 이를 실제로 대조해 검증한다).

    Args (생성자): 상위 클래스와 동일. `ramp_score=None` 이면 V1 이진 게이트로 축퇴한다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rows: List[_Check] = []
        self._checks: Optional[pd.DatetimeIndex] = None
        self._top_n: Optional[int] = None

    # ── 관측 ────────────────────────────────────────────────────────
    def _targets(self, row: np.ndarray, close_row: np.ndarray, held_now: np.ndarray,
                 top_n: int, entry_score: float, exit_score: float
                 ) -> Tuple[np.ndarray, float, np.ndarray]:
        """상위 클래스의 판정을 그대로 쓰고, 그 결과를 기록한다."""
        w_names, w_cash, sel = super()._targets(
            row, close_row, held_now, top_n, entry_score, exit_score)
        _, _, gate = self._ramp_bounds(entry_score)
        # 후보 자격의 전제는 '점수와 가격이 둘 다 있을 것' — 엔진의 elig 판정과 같은 조건.
        usable = ~np.isnan(row) & ~np.isnan(close_row)
        # 자격 문턱은 엔진과 똑같이 히스테리시스로 — 보유는 exit_score, 미보유는 gate. 신규
        # 문턱(gate)만 세면 exit~gate 구간에 잔류한 보유 종목이 슬롯을 차지하는데도 '자격 통과'
        # 에서 빠져, CSV 에 '소비슬롯 > 자격통과' 라는 자기모순 행이 나온다(예: 방어 국면에
        # 신규 자격 0인데 보유 잔류로 슬롯은 채워진 달). 이렇게 하면 자격통과 ≥ 소비슬롯 이 보장된다.
        thr = np.where(held_now, exit_score, gate)
        self._rows.append(_Check(
            fill=1.0 - float(w_cash),          # 못 채운 슬롯 몫이 현금이므로 노출 = 1 − 현금
            slots_used=int(len(sel)),
            n_valid=int(usable.sum()),
            n_gate=int((usable & (row >= thr)).sum()),
        ))
        return w_names, w_cash, sel

    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame,
                  atr: Optional[pd.DataFrame], cash_ret: pd.Series, top_n: int,
                  period: str, trailing, entry_score: float, exit_score: float):
        """상위 클래스 루프를 그대로 돌리고, 기록을 엔진의 체크일에 붙인다.

        루프를 복제하지 않는다. `_targets()` 는 체크일마다 정확히 한 번 불리므로 호출 순서가
        곧 체크일 순서다. 그 가정이 깨지면(엔진 변경 등) 조용히 어긋난 시계열을 내는 대신
        멈춘다 — 이 프로젝트에서 조용한 이탈은 이미 두 번 사고를 냈다.
        """
        self._rows = []
        out = super()._simulate(closes, scores, atr, cash_ret, top_n, period, trailing,
                                entry_score, exit_score)
        checks = closes.index[period_mask(closes.index, period)]
        if len(checks) != len(self._rows):
            raise RuntimeError(
                f"[exposure] 체크일 {len(checks)}개인데 관측 {len(self._rows)}개 — "
                f"엔진의 _targets() 호출 규약이 바뀐 것으로 보입니다. 측정을 신뢰할 수 없습니다.")
        self._checks = checks
        self._top_n = top_n
        return out

    # ── 결과 ────────────────────────────────────────────────────────
    def exposure(self, sleeve_weight: float) -> ExposureResult:
        """관측을 `ExposureResult` 로 낸다.

        Args:
            sleeve_weight: 포트폴리오에서 사테라이트가 갖는 비중(IRPConfig.satellite_weight).
        Raises:
            RuntimeError: 백테스트를 돌리기 전에 부른 경우.
        """
        if self._checks is None or self._top_n is None:
            raise RuntimeError("[exposure] run() 을 먼저 호출해야 합니다.")
        idx = self._checks
        logger.info(f"노출 관측 · 체크 {len(idx)}회 · "
                    f"{idx[0]:%Y-%m-%d}~{idx[-1]:%Y-%m-%d}")
        return ExposureResult(
            fill=pd.Series([r.fill for r in self._rows], index=idx, name="fill"),
            slots_used=pd.Series([r.slots_used for r in self._rows], index=idx, name="slots_used"),
            n_valid=pd.Series([r.n_valid for r in self._rows], index=idx, name="n_valid"),
            n_gate=pd.Series([r.n_gate for r in self._rows], index=idx, name="n_gate"),
            sleeve_weight=float(sleeve_weight),
            top_n=int(self._top_n),
        )
