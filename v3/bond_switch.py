"""[A-2] 채권 슬리브 추세 스위칭 — 30% 채권 몫을 동결 규칙으로 듀레이션 로테이션.

채권 3종(단기채 153130 · 국고3 114260 · 국고30 439870)에 **동결 TrendScore·경사 규칙을 그대로**
적용해, 30% 안에서 점수 상위 듀레이션에 배분한다(새 규칙 발명 금지 — 사테라이트와 같은
`SatelliteBacktesterV2` 를 문턱 52·경사 52→60·바닥 0.3·ramp_hold 로 재사용). 못 채운 몫은 현금
대용 단기채(153130)로 간다 → 장기채 점수가 문턱 아래로 내려가는 금리 급등기엔 자동으로 단기채로
회귀한다(관전점).

구조: 사테라이트 70% 슬리브와 **채권 30% 슬리브** 두 합성 자산을 만들어 분기 + 임계로 리밸런싱한다.
원본 IRP 는 채권을 고정비중으로만 다뤄 이 자리가 없으므로 run() 을 재정의한다(그 안에서 부모의
`_simulate`·`_benchmark`·`_to_result` 는 그대로 재사용 — 리밸런싱·벤치·포장 로직 중복 없음).

주의: 439870(국고30)이 2022-08 상장이라 두 슬리브 공통 거래일이 2022-08 부터다 → 이 실험은
전체 창(2020~)이 아니라 절단 창에서만 측정된다(A-1 과 같은 데이터 한계).
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import pandas as pd

from indicator import Indicator
from irp.backtester_v2 import IRPBacktesterV2
from irp.config import IRPConfig
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2
from satellite.config import SatelliteConfig

logger = logging.getLogger(__name__)

_SAT_KEY = "__SAT__"
_BOND_KEY = "__BOND__"


class BondSwitchBacktester(IRPBacktesterV2):
    """채권 30% 를 듀레이션 추세 로테이션으로 굴리는 IRP 백테스터.

    Args (생성자):
        bond_universe: 로테이션 후보 채권 티커(듀레이션 사다리). 기본 [153130,114260,439870].
        cash_ticker: 빈 슬롯 대피처(기본 153130 단기채 = 금리 급등기 회귀처).
        나머지: `IRPBacktesterV2` 와 동일(satellite 는 동결 Tier 2-a 슬리브를 주입).
    """

    def __init__(self, loader, indicator: Indicator, cost: float = 0.0010, *,
                 satellite=None, allow_missing: bool = False,
                 bond_universe: Optional[List[str]] = None,
                 cash_ticker: str = "153130",
                 bond_ramp_score: Optional[float] = None,
                 bond_check_period: Optional[str] = None):
        super().__init__(loader=loader, indicator=indicator, cost=cost,
                         satellite=satellite, allow_missing=allow_missing)
        self.bond_universe = list(bond_universe or ["153130", "114260", "439870"])
        self.bond_cash = cash_ticker
        self._bond_indicator = indicator
        # 스위칭 손잡이(A-2 plateau 측정용). None 이면 현행 spec 으로 축퇴:
        #   문턱 = FROZEN_RAMP 의 ramp_score(52) · 주기 = icfg 사테라이트 체크주기(M).
        self.bond_ramp_score = bond_ramp_score
        self.bond_check_period = bond_check_period

    def _bond_sleeve_cfg(self, icfg: IRPConfig) -> SatelliteConfig:
        """채권 슬리브용 사테라이트 설정(동결 문턱·top_n=후보수·현금=단기채)."""
        return SatelliteConfig(
            name="채권 듀레이션 로테이션",
            check_period=self.bond_check_period or icfg.satellite.check_period,  # 주기 손잡이(기본 M)
            top_n=len(self.bond_universe),              # 3종 전부 보유 가능(경사로 크기 조절)
            entry_score=60.0, exit_score=45.0,          # 동결 게이트(경사가 실제 크기를 정함)
            cash_ticker=self.bond_cash,
            universe=list(self.bond_universe),
            names=dict(icfg.satellite.names),
        )

    def _required_codes(self, icfg: IRPConfig) -> List[str]:
        """무결성 가드 대상: 사테라이트 유니버스 + 채권 후보 + 현금 + 벤치마크."""
        s = icfg.satellite
        return [*s.universe, s.cash_ticker, *self.bond_universe, self.bond_cash,
                icfg.benchmark_ticker]

    def run(self, icfg: IRPConfig, start=None, end=None):
        self.guard.check(self._required_codes(icfg), allow_missing=self.allow_missing)

        # (1) 70% 사테라이트 슬리브(동결 Tier 2-a) — 주입된 self.satellite 그대로.
        sat = self.satellite.run(icfg.satellite, start=start, end=end, trailing=None)
        # (2) 30% 채권 슬리브 — 동결 경사 규칙을 채권 후보에 재사용.
        lo, full, floor = FROZEN_RAMP
        if self.bond_ramp_score is not None:  # 문턱 손잡이만 교체(만충점수·바닥은 동결 고정)
            lo = self.bond_ramp_score
        bond_bt = SatelliteBacktesterV2(
            loader=self.loader, indicator=self._bond_indicator, cost=self.cost,
            ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
        bond = bond_bt.run(self._bond_sleeve_cfg(icfg), start=start, end=end, trailing=None)

        # (3) 두 합성 자산(사테라이트·채권 슬리브)을 분기 + 임계로 리밸런싱(부모 로직 재사용).
        panel = pd.DataFrame({_SAT_KEY: sat.equity, _BOND_KEY: bond.equity}).sort_index()
        panel = panel.loc[start:end].dropna()
        if len(panel) < 2:
            raise ValueError("[bond_switch] 사테라이트·채권 슬리브 공통 거래일 부족.")
        rets = panel.pct_change(fill_method=None).fillna(0.0)
        weights: Dict[str, float] = {_SAT_KEY: icfg.satellite_weight,
                                     _BOND_KEY: icfg.bond_weight}
        equity, rb_dates = self._simulate(rets, weights, icfg.rebalance_period,
                                          icfg.rebalance_threshold)
        benchmark, bench_name = self._benchmark(icfg, rets, weights)
        logger.info(f"채권 추세 스위칭 · {panel.index[0]:%Y-%m-%d}~{panel.index[-1]:%Y-%m-%d} · "
                    f"채권 로테이션 {len(bond.rotations_log or [])}구간 · 리밸런싱 {len(rb_dates)}회")
        return self._to_result(icfg, rets.index, equity, benchmark, bench_name,
                               rb_dates, sat.rotations_log)
