"""[C-1] IRP 사테라이트 슬리브에 트레일링 스탑을 켠다(원본·V2 무수정).

원본 `IRPBacktester.run` 은 70% 슬리브를 항상 `trailing=None` 으로 돌린다(휴가 전 체크리스트에서
IRP 는 스탑을 끈 채 동결됐다). 이 실험은 그 한 줄만 바꾸고 싶은데, run() 은 길고 훅 자리가 없다.
그래서 **슬리브 쪽에서** 주입한다: `run(trailing=None)` 호출을 받아도 미리 정해둔 트레일링으로
바꿔치는 얇은 서브클래스. 이렇게 하면 IRP 오케스트레이션(채권 30% + 분기 리밸런싱)은 원본
그대로 두고, 사테라이트 내부에만 스탑이 걸린다.

관전점(전달문): 게이트(진입 52/청산 45)와 트레일링 스탑이 **기능 중복**인가. 게이트는 이미
약세 종목을 슬롯에서 빼 현금(단기채)으로 대피시킨다. 스탑이 추가 방어를 주는지, 아니면
추세장에서 조기 청산으로 상승만 깎는지가 결과로 드러난다.
"""
from __future__ import annotations

import logging
from typing import Optional

from backtest import BacktestResult
from satellite.backtester_v2 import SatelliteBacktesterV2
from satellite.config import SatelliteConfig
from satellite.trailing import TrailingStop

logger = logging.getLogger(__name__)


class TrailingSleeveV2(SatelliteBacktesterV2):
    """동결 Tier 2-a 경사 슬리브 + 강제 트레일링 스탑.

    상위(IRP)가 `trailing=None` 으로 호출해도 생성자에서 받은 스탑을 대신 쓴다. 경사 진입
    파라미터(ramp_score/full_score/ramp_floor/ramp_hold)는 동결값 그대로 위임한다.

    Args (생성자):
        trailing: 강제로 적용할 트레일링 스탑 규칙(예: `AtrTrailingStop`).
        나머지: `SatelliteBacktesterV2` 와 동일.
    """

    def __init__(self, loader, indicator, cost: float = 0.0010, *,
                 trailing: TrailingStop, **ramp_kwargs):
        super().__init__(loader=loader, indicator=indicator, cost=cost, **ramp_kwargs)
        self._forced_trailing = trailing

    def run(self, scfg: SatelliteConfig, start=None, end=None,
            trailing: Optional[TrailingStop] = None) -> BacktestResult:
        # 상위가 넘긴 trailing(항상 None)을 무시하고 강제 트레일링을 쓴다.
        return super().run(scfg, start=start, end=end, trailing=self._forced_trailing)
