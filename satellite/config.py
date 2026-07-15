"""사테라이트(모멘텀 로테이션) 설정 (``config/satellite.json`` 로딩).

후보 종목 리스트 중 **지표 점수 상위 top_n 개를 동일가중 보유**하고, 체크주기마다 재평가해
종목을 교체하는 전략의 정의를 읽는다. 코드 수정 없이 JSON 만 바꿔 지표·주기·개수·유니버스를
조정하고, ``enabled=false`` 로 전체를 끌 수 있다.

기본값: 지표=TrendScore, 체크주기=Month(월간), top_n=7, 유니버스=미국 섹터·원자재·대체/테마 41종목.
매 체크에서 상위 7 슬롯을 채우되, 자격(유효 점수) 종목이 7개 미만이거나 트레일링 스탑으로
청산된 슬롯은 **현금 대용(BIL)** 으로 보유한다. 트레일링 스탑 파라미터도 여기서 조정한다.
(백테스트상 이 유니버스에선 월간 체크가 일간/주간/분기보다 성과가 좋았다.)
JSON 이 없거나 일부 키가 빠져도 dataclass 기본값으로 폴백한다. ``_`` 로 시작하는 키는 무시(주석용).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

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
class TrailingStopConfig:
    """트레일링 스탑 파라미터(고정 방식 비교 + ATR 다이나믹 옵션).

    실제 청산 규칙 객체(`TrailingStop`)는 이 파라미터로 오케스트레이터(main)가 만든다.
    설정은 값만 담고 동작은 규칙 클래스에 맡겨 계층(설정 ↔ 로직)을 분리한다.

    기본 비교는 **고정 비율 여러 개**(예 15% vs 7%)를 헤드투헤드로 돌린다. ATR 다이나믹은
    규칙 클래스로 남겨 두어(atr_period/atr_mult) 필요 시 코드에서 끼워 쓸 수 있다.

    Attributes:
        fixed_pcts: 비교할 고정 후퇴 비율 목록(0.15 = 고점 −15%에서 청산). 각 값마다 1개 변형.
        atr_period: ATR 평활 기간(다이나믹 방식, 옵션).
        atr_mult: ATR 배수(다이나믹, 옵션). 손절가 = 고점 − atr_mult×ATR.
    """
    fixed_pcts: List[float] = field(default_factory=lambda: [0.15, 0.07])
    atr_period: int = 22
    atr_mult: float = 3.0


@dataclass
class SatelliteConfig:
    """사테라이트 로테이션 전략 정의.

    Attributes:
        enabled: 전략 실행 여부(파이프라인 on/off).
        name: 표시명(리포트·로그).
        indicator: 순위 산정 지표 이름(현재 "trend_score" 만 지원).
        trend_score: TrendScore 지표 파라미터(순위 계산에 사용).
        check_period: 재평가 주기. "D"(매 거래일)·"W"·"M"·"Q"·"<N>D".
        top_n: 매 체크에서 채울 상위 슬롯 수(동일가중). 자격 종목이 모자라거나
               트레일링 스탑으로 비면 그 슬롯은 현금 대용(cash_ticker)으로 채운다.
        entry_score: 신규 진입 TrendScore 하한(이 점수 이상만 새로 편입). 미달 슬롯은 현금.
        exit_score: 보유 유지 TrendScore 하한(이 점수 미만이면 청산). entry_score 와의 간격이
                    히스테리시스(잦은 교체 억제) — 진입 60·청산 45면 45~60 구간은 보유만 유지.
        cash_ticker: 빈 슬롯·손절 대피 자금을 담을 현금 대용 티커(기본 BIL, 초단기 국채 ETF).
        trailing: 트레일링 스탑 파라미터(고정 비율 비교 + ATR 옵션).
        universe: 후보 종목 티커 리스트.
        names: {티커: 표시명} 옵션 맵(로테이션 내역 리포트 가독성용). 비면 티커 그대로 표시.
    """
    enabled: bool = True
    name: str = "Satellite"
    indicator: str = "trend_score"
    trend_score: TrendScoreParams = field(default_factory=TrendScoreParams)
    check_period: str = "M"
    top_n: int = 7
    entry_score: float = 60.0
    exit_score: float = 45.0
    cash_ticker: str = "BIL"
    trailing: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    universe: List[str] = field(default_factory=lambda: list(_DEFAULT_UNIVERSE))
    names: Dict[str, str] = field(default_factory=dict)

    def label(self, code: str) -> str:
        """티커의 리포트 표시 라벨(이름이 있으면 ``코드·이름``, 없으면 코드)."""
        nm = self.names.get(code)
        return f"{code}·{nm}" if nm else code

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
        cfg.entry_score = float(raw.get("entry_score", cfg.entry_score))
        cfg.exit_score = float(raw.get("exit_score", cfg.exit_score))
        cfg.cash_ticker = str(raw.get("cash_ticker", cfg.cash_ticker)).strip().upper()
        universe = raw.get("universe")
        if universe:
            cfg.universe = [str(t).strip().upper() for t in universe]
        names = _strip_comments(raw.get("names", {}))
        if names:
            cfg.names = {str(k).strip().upper(): str(v) for k, v in names.items()}

        # 지표 파라미터(부분만 적어도 기본값과 병합)
        ts = _strip_comments(raw.get("trend_score", {}))
        base = TrendScoreParams()
        for k, v in ts.items():
            if hasattr(base, k):
                setattr(base, k, v)
        cfg.trend_score = base

        # 트레일링 스탑 파라미터(부분만 적어도 기본값과 병합)
        tr = _strip_comments(raw.get("trailing", {}))
        tcfg = TrailingStopConfig()
        for k, v in tr.items():
            if hasattr(tcfg, k):
                setattr(tcfg, k, v)
        cfg.trailing = tcfg

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
        if self.entry_score < self.exit_score:
            raise ValueError(f"[satellite] entry_score({self.entry_score})는 exit_score"
                             f"({self.exit_score}) 이상이어야 합니다(히스테리시스).")
        if not self.cash_ticker:
            raise ValueError("[satellite] cash_ticker 가 비어 있습니다(현금 대용 티커를 지정).")
        t = self.trailing
        if not t.fixed_pcts:
            raise ValueError("[satellite] trailing.fixed_pcts 가 비어 있습니다(비교할 비율을 1개 이상).")
        for p in t.fixed_pcts:
            if not (0.0 < p < 1.0):
                raise ValueError(f"[satellite] trailing.fixed_pcts 값은 (0,1) 범위여야 합니다: {p}")
        if t.atr_mult <= 0:
            raise ValueError(f"[satellite] trailing.atr_mult 는 0 보다 커야 합니다: {t.atr_mult}")
