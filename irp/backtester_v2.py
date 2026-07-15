"""IRP 백테스터 V2 — 사테라이트 슬리브 백테스터를 주입 가능하게 한 변형.

원본(`IRPBacktester`)은 생성자에서 `SatelliteBacktester` 를 **직접 만든다**. 그래서 70% 슬리브만
변형(예: 경사 진입 `SatelliteBacktesterV2`)해서 끼워 넣을 자리가 없다. 원본을 고치지 않고
그 자리를 열어 주는 것이 이 클래스의 전부다 — 채권 30% + 분기/임계 리밸런싱 로직은 원본을
그대로 상속해 쓴다(중복 없음).
"""
from __future__ import annotations

import logging
from typing import Optional

from data import DataLoader
from indicator import Indicator
from satellite import SatelliteBacktester

from .backtester import IRPBacktester

logger = logging.getLogger(__name__)


class IRPBacktesterV2(IRPBacktester):
    """사테라이트 슬리브 백테스터를 외부에서 주입할 수 있는 IRP 백테스터.

    Args (생성자):
        loader / indicator / cost: 원본과 동일.
        satellite: 70% 슬리브를 돌릴 백테스터. None 이면 원본이 만든 기본
            `SatelliteBacktester`(= V1 동작)를 그대로 쓴다.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010,
                 satellite: Optional[SatelliteBacktester] = None):
        super().__init__(loader=loader, indicator=indicator, cost=cost)
        if satellite is not None:
            self.satellite = satellite
