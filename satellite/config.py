"""사테라이트(모멘텀 로테이션) 설정 (``config/satellite.json`` 로딩).

후보 종목 리스트 중 **지표 점수 상위 top_n 개를 동일가중 보유**하고, 체크주기마다 재평가해
종목을 교체하는 전략의 정의를 읽는다. 코드 수정 없이 JSON 만 바꿔 지표·주기·개수·유니버스를
조정하고, ``enabled=false`` 로 전체를 끌 수 있다.

기본값: 지표=TrendScore, 체크주기=Month(월간), top_n=4, 유니버스=미국 섹터·원자재·대체/테마 41종목.
(백테스트상 이 유니버스에선 월간 체크가 일간/주간/분기보다 성과가 좋았다.)
JSON 이 없거나 일부 키가 빠져도 dataclass 기본값으로 폴백한다. ``_`` 로 시작하는 키는 무시(주석용).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List

from config import TrendScoreParams  # 지표 파라미터 재사용(단일종목 전략과 동일 정의)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_JSON = os.path.join(_ROOT, "config", "satellite.json")

# 기본 후보 유니버스(미국 섹터 + 원자재 + 대체/테마)
_DEFAULT_UNIVERSE: List[str] = [
    # US Sectors
    "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "SMH", "IBB", "KBE", "XOP", "ITB",
    # Commodities
    "DBC", "PDBC", "IAU", "SLV", "PPLT", "GSG",
    "USO", "UNG", "URA", "CPER", "DBA", "LIT",
    # Alternatives & Thematic
    "REM", "DBMF", "KMLM", "WTMF", "PSP", "UUP", "FXE",
    "IGV", "CIBR", "BOTZ", "TAN", "PAVE", "ARKK",
]


def _strip_comments(d: dict) -> dict:
    """``_`` 로 시작하는 문서용 메타 키를 제거한다."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


@dataclass
class SatelliteConfig:
    """사테라이트 로테이션 전략 정의.

    Attributes:
        enabled: 전략 실행 여부(파이프라인 on/off).
        name: 표시명(리포트·로그).
        indicator: 순위 산정 지표 이름(현재 "trend_score" 만 지원).
        trend_score: TrendScore 지표 파라미터(순위 계산에 사용).
        check_period: 재평가 주기. "D"(매 거래일)·"W"·"M"·"Q"·"<N>D".
        top_n: 매 체크에서 보유할 상위 종목 수(동일가중).
        universe: 후보 종목 티커 리스트.
    """
    enabled: bool = True
    name: str = "Satellite"
    indicator: str = "trend_score"
    trend_score: TrendScoreParams = field(default_factory=TrendScoreParams)
    check_period: str = "M"
    top_n: int = 4
    universe: List[str] = field(default_factory=lambda: list(_DEFAULT_UNIVERSE))

    # ── 로딩 ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> "SatelliteConfig":
        """``config/satellite.json`` 을 읽어 `SatelliteConfig` 로 만든다.

        파일이 없으면 ``enabled=False`` 인 빈 설정을 돌려줘 파이프라인이 조용히 건너뛰게 한다.
        """
        path = path or _DEFAULT_JSON
        if not os.path.exists(path):
            logger.info(f"{path} 없음 → 사테라이트 백테스트 생략")
            return cls(enabled=False)

        with open(path, encoding="utf-8") as f:
            raw = _strip_comments(json.load(f))

        cfg = cls()
        cfg.enabled = bool(raw.get("enabled", True))
        cfg.name = str(raw.get("name", cfg.name))
        cfg.indicator = str(raw.get("indicator", cfg.indicator)).lower()
        cfg.check_period = str(raw.get("check_period", cfg.check_period))
        cfg.top_n = int(raw.get("top_n", cfg.top_n))
        universe = raw.get("universe")
        if universe:
            cfg.universe = [str(t).strip().upper() for t in universe]

        # 지표 파라미터(부분만 적어도 기본값과 병합)
        ts = _strip_comments(raw.get("trend_score", {}))
        base = TrendScoreParams()
        for k, v in ts.items():
            if hasattr(base, k):
                setattr(base, k, v)
        cfg.trend_score = base

        if cfg.enabled:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        """설정 무결성 검사(치명은 예외, 경미한 이상은 경고)."""
        if self.indicator != "trend_score":
            raise ValueError(f"[satellite] 지원하지 않는 지표: {self.indicator} (현재 'trend_score' 만 지원)")
        if not self.universe:
            raise ValueError("[satellite] universe 가 비어 있습니다(후보 종목을 1개 이상 지정).")
        if self.top_n < 1:
            raise ValueError(f"[satellite] top_n 은 1 이상이어야 합니다: {self.top_n}")
        if self.top_n > len(self.universe):
            logger.warning(f"top_n({self.top_n})이 유니버스 크기({len(self.universe)})보다 큽니다 "
                           "→ 가용 종목 전부 보유합니다.")
