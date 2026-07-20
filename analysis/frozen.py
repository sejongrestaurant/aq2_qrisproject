"""동결 구성 조립 한곳 — 분석 러너들이 '같은 상품'을 재도록 보장한다.

분석 러너가 늘어나면서 각자 백테스터를 조립하게 됐다(`run_segments` · `run_exposure` · …).
조립이 복붙되면 한쪽만 바뀐 채 다른 쪽이 남아 **서로 다른 상품의 수치가 같은 제안서에**
실린다. 이 프로젝트에서 조용한 이탈은 이미 여러 번 사고를 냈으므로(전달 체인 오염·창 불일치),
동결 V2 와 V1 기준선의 조립을 여기 한 함수로 모은다.

동결값(`FROZEN_RAMP`) 자체는 여기서 정하지 않는다 — 그건 실험 러너(`run_v2.py`)의 것이고,
여기로 복제하면 동결값이 두 곳에 생긴다. 러너가 인자로 넘긴다.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from data import DataLoader
from indicator import Indicator, TrendScoreIndicator
from irp.backtester_v2 import IRPBacktesterV2
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger(__name__)

Ramp = Optional[Tuple[float, float, float]]


def build_sleeve(loader: DataLoader, cost: float, ramp: Ramp, *,
                 indicator: Optional[Indicator] = None,
                 sleeve_cls=SatelliteBacktesterV2, **kwargs) -> Optional[SatelliteBacktesterV2]:
    """70% 사테라이트 슬리브를 만든다. `ramp=None` 이고 기본 클래스면 None(= V1 슬리브).

    Args:
        cost: 슬리브 내부 로테이션에 물릴 왕복 거래비용.
        ramp: (경사하단, 만충점수, 하한충전율). None 이면 이진 게이트로 축퇴.
        sleeve_cls: 슬리브 클래스. 관측 프로브(`ExposureProbe`)를 끼울 때 바꾼다.
        kwargs: 슬리브 클래스에 그대로 넘길 추가 인자.
    """
    if ramp is None and sleeve_cls is SatelliteBacktesterV2 and not kwargs:
        return None                      # 미주입 → IRPBacktesterV2 가 원본 V1 슬리브를 쓴다
    kw = dict(kwargs)
    if ramp is not None:
        lo, full, floor = ramp
        kw.update(ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    return sleeve_cls(loader=loader, indicator=indicator or TrendScoreIndicator(),
                      cost=cost, **kw)


def build_irp(loader: DataLoader, cost: float, ramp: Ramp, *,
              indicator: Optional[Indicator] = None,
              sleeve_cost: Optional[float] = None,
              allow_missing: bool = False,
              sleeve_cls=SatelliteBacktesterV2, **kwargs) -> IRPBacktesterV2:
    """동결 구성(또는 V1 기준선) IRP 백테스터를 만든다.

    Args:
        cost: **상위(분기 리밸런싱)** 왕복 거래비용.
        sleeve_cost: 슬리브 내부 로테이션 비용. None 이면 `cost` 와 같게 둔다.
            회전율 측정에서 두 계층의 비용을 따로 껐다 켜기 위해 분리해 뒀다.
        ramp / sleeve_cls / kwargs: `build_sleeve()` 로 전달.
        allow_missing: 유니버스 결손 허용(러너 플래그 전용).
    """
    ind = indicator or TrendScoreIndicator()
    sat = build_sleeve(loader, cost if sleeve_cost is None else sleeve_cost, ramp,
                       indicator=ind, sleeve_cls=sleeve_cls, **kwargs)
    return IRPBacktesterV2(loader=loader, indicator=ind, cost=cost, satellite=sat,
                           allow_missing=allow_missing)
