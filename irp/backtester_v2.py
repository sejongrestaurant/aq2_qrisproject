"""IRP 백테스터 V2 — 사테라이트 슬리브 주입 + 유니버스 무결성 사전 점검.

원본(`IRPBacktester`)에 두 가지가 없어서 이 클래스가 존재한다(원본 무수정 원칙):

1. **슬리브 교체 자리** — 원본은 생성자에서 `SatelliteBacktester` 를 직접 만든다. 그래서 70%
   슬리브만 변형(예: 경사 진입 `SatelliteBacktesterV2`)해 끼워 넣을 수 없다.
2. **fail-loud 가드** — 원본의 로딩 경로는 실패해도 멈추지 않는다(종목은 후보에서 빼고 경고,
   현금 대용은 무이자 현금으로, 벤치마크는 드리프트로 폴백). 그 결과 **설정과 다른 전략이
   조용히 돌아가고 결과는 그럴듯하게 나온다**. 실행 전에 필수 종목을 전부 읽어 보고 하나라도
   실패하면 멈춘다. 자세한 배경은 `data/integrity.py` 참조.

채권 30% + 분기/임계 리밸런싱 로직은 원본을 그대로 상속해 쓴다(중복 없음).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from data import DataLoader
from data.integrity import UniverseGuard
from indicator import Indicator
from satellite import SatelliteBacktester

from .backtester import IRPBacktester
from .config import IRPConfig

logger = logging.getLogger(__name__)


class IRPBacktesterV2(IRPBacktester):
    """슬리브 주입이 가능하고, 유니버스가 온전할 때만 도는 IRP 백테스터.

    Args (생성자):
        loader / indicator / cost: 원본과 동일.
        satellite: 70% 슬리브를 돌릴 백테스터. None 이면 원본이 만든 기본
            `SatelliteBacktester`(= V1 동작)를 그대로 쓴다.
        allow_missing: True 면 종목이 빠져도 진행한다(경고만). **명시적 의사표시 전용** —
            러너의 `--allow-missing` 플래그로만 켜고, 기본값을 바꾸지 않는다.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010,
                 satellite: Optional[SatelliteBacktester] = None,
                 allow_missing: bool = False):
        super().__init__(loader=loader, indicator=indicator, cost=cost)
        if satellite is not None:
            self.satellite = satellite
        self.allow_missing = bool(allow_missing)
        # 워밍업 봉 수는 지표가 안다(TrendScore.min_len=252). 없으면 봉 수 점검만 건너뛴다.
        self.guard = UniverseGuard(loader, min_bars=getattr(indicator, "min_len", None))

    # ── public ──────────────────────────────────────────────────────
    def run(self, icfg: IRPConfig, start=None, end=None):
        """유니버스 무결성을 확인한 뒤 원본 로직으로 백테스트한다."""
        self.guard.check(self._required_codes(icfg), allow_missing=self.allow_missing)
        return super().run(icfg, start=start, end=end)

    # ── 내부 ────────────────────────────────────────────────────────
    @staticmethod
    def _required_codes(icfg: IRPConfig) -> List[str]:
        """설정이 요구하는 모든 티커.

        후보 유니버스만으로는 부족하다 — 아래 셋도 실패 시 **조용한 폴백** 경로를 갖고 있어
        똑같이 결과를 바꾼다:
          · cash_ticker  — 빈 슬롯 대피처. 실패하면 무이자 현금(수익 0)이 된다. 78개월 중
            62개월이 슬롯 미달이었으므로 이 폴백은 결과를 크게 바꾼다.
          · bonds        — 30% 고정 슬리브(원본은 이것만 치명 처리한다).
          · benchmark    — 실패하면 '무리밸런싱 30/70 드리프트' 라는 다른 벤치마크가 된다.
        """
        s = icfg.satellite
        return [*s.universe, s.cash_ticker, *icfg.bonds, icfg.benchmark_ticker]
