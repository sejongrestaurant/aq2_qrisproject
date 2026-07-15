"""IRP(개인형 퇴직연금) ETF 전략 설정 (``config/irp.json`` 로딩).

**채권 30% 고정 + 사테라이트(섹터/자산 모멘텀 로테이션) 70%** 를 분기마다 목표비중으로
되돌리는 자산배분 전략의 정의를 읽는다. 채권 슬리브는 3종을 각 10%(합 30%)로 고정하고,
나머지 70% 는 월간 Top-N 모멘텀 로테이션(사테라이트)에 맡긴다.

설계상 사테라이트 슬리브는 기존 `SatelliteConfig`/`SatelliteBacktester` 를 그대로 재사용한다
(관심사 분리·중복 방지). IRP 는 그 위에 '채권 바닥 + 분기 리밸런싱' 만 얹는다.

JSON 이 없거나 일부 키가 빠져도 dataclass 기본값으로 폴백한다. ``_`` 로 시작하는 키는 무시(주석용).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from satellite import SatelliteConfig

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_JSON = os.path.join(_ROOT, "config", "irp.json")

# 채권 슬리브(각 10%, 합 30%): 단기채권·국고채3년·종합채권(AA-이상) 액티브
_DEFAULT_BONDS: Dict[str, float] = {"153130": 0.10, "114260": 0.10, "273130": 0.10}

# 티커 → 표시명(로테이션 내역 리포트 가독성용). 채권 3종 + 사테라이트 유니버스 전체.
_DEFAULT_NAMES: Dict[str, str] = {
    # 채권
    "153130": "KODEX 단기채권", "114260": "KODEX 국고채3년",
    "273130": "KODEX 종합채권(AA-이상)액티브",
    # 글로벌 주식
    "379800": "KODEX 미국S&P500", "379810": "KODEX 미국나스닥100",
    "453810": "KODEX 인도Nifty50", "101280": "KODEX 일본TOPIX100",
    "169950": "KODEX 차이나A50", "283580": "KODEX 차이나CSI300",
    "099140": "KODEX 차이나H", "105010": "TIGER 라틴35",
    "195930": "TIGER 유로스탁스50(합성H)",
    # 미국 S&P500 섹터
    "453650": "KODEX 미국S&P500금융", "200030": "KODEX 미국S&P500산업재(합성)",
    "218420": "KODEX 미국S&P500에너지(합성)", "463640": "KODEX 미국S&P500유틸리티",
    "463690": "KODEX 미국S&P500커뮤니케이션", "463680": "KODEX 미국S&P500테크놀로지",
    "453660": "KODEX 미국S&P500경기소비재", "453630": "KODEX 미국S&P500필수소비재",
    "453640": "KODEX 미국S&P500헬스케어",
    # 원자재·리츠
    "160580": "TIGER 구리실물", "0072R0": "TIGER KRX금현물",
    "0189B0": "TIGER 은액티브", "329200": "TIGER 리츠부동산인프라",
    # 한국 섹터
    "117700": "KODEX 건설", "266390": "KODEX 경기소비재", "102960": "KODEX 기계장비",
    "091160": "KODEX 반도체", "140700": "KODEX 보험", "117460": "KODEX 에너지화학",
    "140710": "KODEX 운송", "091170": "KODEX 은행", "091180": "KODEX 자동차",
    "102970": "KODEX 증권", "117680": "KODEX 철강", "266410": "KODEX 필수소비재",
    "266420": "KODEX 헬스케어", "266370": "KODEX IT", "266360": "KODEX K콘텐츠",
}

# 사테라이트 슬리브(70%) 후보 유니버스: 글로벌 주식 · 미국 섹터 · 원자재/리츠 · 한국 섹터
_DEFAULT_UNIVERSE: List[str] = [
    # 글로벌 주식
    "379800", "379810", "453810", "101280", "169950", "283580", "099140", "105010", "195930",
    # 미국 S&P500 섹터
    "453650", "200030", "218420", "463640", "463690", "463680", "453660", "453630", "453640",
    # 원자재·리츠
    "160580", "0072R0", "0189B0", "329200",
    # 한국 섹터
    "117700", "266390", "102960", "091160", "140700", "117460", "140710", "091170",
    "091180", "102970", "117680", "266410", "266420", "266370", "266360",
]


def _strip_comments(d: dict) -> dict:
    """``_`` 로 시작하는 문서용 메타 키를 제거한다."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _default_satellite() -> SatelliteConfig:
    """IRP 70% 슬리브용 사테라이트 기본 설정(순수 Top-N 모멘텀 로테이션).

    IRP 사양은 체크주기 M·개수 7·유니버스만 지정하므로, 점수 게이트는 끄고(entry=exit=0 →
    유효 종목이면 항상 상위 7 편입) 트레일링 스탑도 쓰지 않는다(백테스터 호출 시 trailing=None).
    빈 슬롯이 생기면 담을 현금 대용은 단기채권(153130)으로 둔다(유니버스가 커 실제로는 드묾).
    """
    return SatelliteConfig(
        name="IRP 사테라이트 슬리브",
        check_period="M",
        top_n=7,
        entry_score=0.0,      # 점수 게이트 사실상 끔(항상 상위 7 채움)
        exit_score=0.0,
        cash_ticker="153130",
        universe=list(_DEFAULT_UNIVERSE),
        names=dict(_DEFAULT_NAMES),  # 로테이션 내역 리포트에 한글명 표시
    )


@dataclass
class IRPConfig:
    """IRP ETF 전략 정의(채권 고정 + 사테라이트 로테이션).

    Attributes:
        enabled: 전략 실행 여부(파이프라인 on/off).
        name: 표시명(리포트·로그).
        rebalance_period: 상위 리밸런싱 주기(채권/사테라이트 비중 복원). "Q"(분기) 기본.
        rebalance_threshold: 드리프트 임계(사테라이트 비중이 목표에서 이 값을 넘게 이탈하면 리밸런싱).
            예 0.07 = ±7%p. None/0 이면 임계 리밸런싱 끔(주기만). 주기와 함께 켜면 둘 다 트리거.
        bonds: {채권티커: 비중}. 합이 채권 총배분(기본 0.30). 사테라이트 비중 = 1 − 합.
        satellite: 70% 슬리브 사테라이트 설정(유니버스·체크주기·top_n).
        start / end: IRP 전용 백테스트 구간 override. None 이면 파이프라인 공통 구간을 따른다.
            글로벌·미국 섹터 ETF 상장이 늦어(2021~2023) 전체 유니버스가 갖춰지는 2020년 이후로
            시작해야 로테이션이 대표성을 갖는다(기본 start=2020-01-01).
    """
    enabled: bool = True
    name: str = "IRP 섹터로테이션"
    rebalance_period: str = "Q"
    rebalance_threshold: Optional[float] = None
    bonds: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_BONDS))
    satellite: SatelliteConfig = field(default_factory=_default_satellite)
    start: Optional[str] = "2020-01-01"
    end: Optional[str] = None

    @property
    def bond_weight(self) -> float:
        """채권 슬리브 총비중(bonds 비중 합)."""
        return float(sum(self.bonds.values()))

    @property
    def satellite_weight(self) -> float:
        """사테라이트 슬리브 비중(= 1 − 채권 총비중)."""
        return 1.0 - self.bond_weight

    # ── 로딩 ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> "IRPConfig":
        """``config/irp.json`` 을 읽어 `IRPConfig` 로 만든다.

        파일이 없으면 ``enabled=False`` 인 빈 설정을 돌려줘 파이프라인이 조용히 건너뛰게 한다.
        """
        path = path or _DEFAULT_JSON
        if not os.path.exists(path):
            logger.info(f"{path} 없음 → IRP 백테스트 생략")
            return cls(enabled=False)

        with open(path, encoding="utf-8") as f:
            raw = _strip_comments(json.load(f))

        cfg = cls()
        cfg.enabled = bool(raw.get("enabled", True))
        cfg.name = str(raw.get("name", cfg.name))
        cfg.rebalance_period = str(raw.get("rebalance_period", cfg.rebalance_period)).strip().upper()
        thr = raw.get("rebalance_threshold")
        cfg.rebalance_threshold = float(thr) if thr not in (None, "", 0, 0.0, False) else None
        if "start" in raw:
            cfg.start = raw.get("start") or None
        if "end" in raw:
            cfg.end = raw.get("end") or None

        bonds = _strip_comments(raw.get("bonds", {}))
        if bonds:
            cfg.bonds = {str(t).strip().upper(): float(w) for t, w in bonds.items()}

        # 사테라이트 슬리브: 기본값을 만든 뒤 JSON 의 지정 키만 덮어쓴다(부분 지정 허용).
        sat = _strip_comments(raw.get("satellite", {}))
        scfg = _default_satellite()
        scfg.check_period = str(sat.get("check_period", scfg.check_period))
        scfg.top_n = int(sat.get("top_n", scfg.top_n))
        scfg.entry_score = float(sat.get("entry_score", scfg.entry_score))
        scfg.exit_score = float(sat.get("exit_score", scfg.exit_score))
        scfg.cash_ticker = str(sat.get("cash_ticker", scfg.cash_ticker)).strip().upper()
        universe = sat.get("universe")
        if universe:
            scfg.universe = [str(t).strip().upper() for t in universe]
        names = _strip_comments(sat.get("names", {}))
        if names:  # JSON 에 명시하면 기본 한글명 위에 덮어쓴다(부분 지정 허용)
            scfg.names = {**scfg.names, **{str(k).strip().upper(): str(v) for k, v in names.items()}}
        cfg.satellite = scfg

        if cfg.enabled:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        """설정 무결성 검사(치명은 예외, 경미한 이상은 경고)."""
        if not self.bonds:
            raise ValueError("[irp] bonds 가 비어 있습니다(채권 티커:비중을 1개 이상 지정).")
        if not (0.0 < self.bond_weight < 1.0):
            raise ValueError(f"[irp] 채권 총비중은 (0,1) 범위여야 합니다: {self.bond_weight}")
        if self.rebalance_period in ("", "NONE", "OFF") and not self.rebalance_threshold:
            raise ValueError("[irp] 리밸런싱 트리거가 없습니다(rebalance_period 또는 "
                             "rebalance_threshold 중 하나는 지정).")
        if self.rebalance_threshold is not None and not (0.0 < self.rebalance_threshold < 1.0):
            raise ValueError(f"[irp] rebalance_threshold 는 (0,1) 범위여야 합니다: "
                             f"{self.rebalance_threshold}")
        # 사테라이트 슬리브도 자체 규칙으로 검증(유니버스·top_n 등).
        self.satellite.validate()
