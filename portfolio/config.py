"""포트폴리오 설정 (``config/portfolio.json`` 로딩).

여러 종목을 **비중대로 동시 보유**하는 자산배분 포트폴리오의 정의를 프로젝트 루트의
``config/portfolio.json`` 에서 읽는다. 코드 수정 없이 JSON 만 바꿔 종목·비중·리밸런싱을
조정하고, 여러 단계에서 기능을 끌 수 있다:

    · portfolio.enabled = false        → 포트폴리오 백테스트 자체를 건너뜀
    · rebalance.enabled = false        → 리밸런싱 없이 초기 비중 매수후보유(드리프트 허용)
    · rebalance.period  = "none"/null  → 주기 리밸런싱만 끔(임계 리밸런싱은 유지)
    · rebalance.threshold = 0/null     → 임계(드리프트) 리밸런싱만 끔(주기 리밸런싱은 유지)

JSON 이 없거나 일부 키가 빠져도 dataclass 기본값으로 폴백한다(부분만 적어도 동작).
``_`` 로 시작하는 키(``_comment`` 등)는 문서용 메타로 간주해 무시한다.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# portfolio/ 의 부모(프로젝트 루트) 기준으로 기본 설정 경로를 잡는다.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_JSON = os.path.join(_ROOT, "config", "portfolio.json")

# 주기 코드 → 사람이 읽는 라벨(리포트 표시용)
_PERIOD_LABEL = {"M": "월간", "Q": "분기", "Y": "연간", "A": "연간", "W": "주간"}


def _strip_comments(d: dict) -> dict:
    """``_`` 로 시작하는 문서용 메타 키를 제거한다."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


@dataclass
class RebalanceConfig:
    """리밸런싱 규칙.

    Attributes:
        enabled: 리밸런싱 전체 on/off. False 면 초기 비중 그대로 매수후보유(드리프트 허용).
        period: 주기 리밸런싱 간격. "M"(월)·"Q"(분기)·"Y"(연)·"W"(주)·"<N>D"(N거래일)·
               None(주기 리밸런싱 끔). 각 주기의 첫 거래일에 목표비중으로 복원한다.
        threshold: 드리프트 임계(비중의 절대편차). 어떤 종목이든 목표 대비 이 값을 넘게
                  벗어나면 리밸런싱한다. 예 0.05 = 5%p. None 이면 임계 리밸런싱 끔.
    """
    enabled: bool = True
    period: Optional[str] = "Q"
    threshold: Optional[float] = 0.05

    def describe(self) -> str:
        """사람이 읽는 리밸런싱 요약(리포트 라벨용). 예: '분기 · ±5% 리밸런싱'."""
        if not self.enabled:
            return "리밸런싱 없음"
        parts = []
        if self.period:
            parts.append(_PERIOD_LABEL.get(self.period)
                         or (f"{self.period[:-1]}거래일" if self.period.endswith("D") else self.period))
        if self.threshold:
            parts.append(f"±{self.threshold * 100:.0f}%")
        return (" · ".join(parts) + " 리밸런싱") if parts else "리밸런싱(트리거 없음)"


@dataclass
class WeightingConfig:
    """비중 틸트 규칙(TrendScore 기반 시그모이드).

    각 종목의 기본비중에 지표점수로 만든 배수를 곱한 뒤 합=1 로 재정규화한다. 배수는
    시그모이드로 ``min~max`` 사이를 움직이며, 점수가 ``center`` 일 때 중앙값(=(min+max)/2)이다.

    Attributes:
        method: "none"(고정비중) | "trend_score_sigmoid"(TrendScore 시그모이드 틸트).
        center: 시그모이드 중심 점수(TrendScore 0~100 기준, 기본 45).
        min_mult / max_mult: 배수 하한/상한(기본 0.3 ~ 3.0).
        steepness: 시그모이드 기울기. 클수록 점수 변화에 배수가 급변(기본 0.1 → 0~100 전 구간 활용).
    """
    method: str = "none"
    center: float = 45.0
    min_mult: float = 0.3
    max_mult: float = 3.0
    steepness: float = 0.1

    @property
    def enabled(self) -> bool:
        """틸트 사용 여부(method 가 none/빈값이 아니면 True)."""
        return self.method not in ("none", None, "")

    def describe(self) -> str:
        """사람이 읽는 요약(리포트 라벨용)."""
        if not self.enabled:
            return "고정비중"
        return f"TS가중(중심{self.center:.0f}·{self.min_mult:.1f}~{self.max_mult:.1f})"


@dataclass
class PortfolioConfig:
    """자산배분 포트폴리오 정의.

    Attributes:
        enabled: 포트폴리오 백테스트 실행 여부(파이프라인 on/off).
        name: 표시명(리포트·로그).
        holdings: {티커: 목표비중}. 합이 1.0 이 되도록 로딩 시 정규화된다.
        rebalance: 리밸런싱 규칙.
        weighting: 비중 틸트 규칙(TrendScore 시그모이드). 기본은 고정비중(none).
    """
    enabled: bool = True
    name: str = "Portfolio"
    holdings: Dict[str, float] = field(default_factory=dict)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)

    # ── 로딩 ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> "PortfolioConfig":
        """``config/portfolio.json`` 을 읽어 `PortfolioConfig` 로 만든다.

        파일이 없으면 ``enabled=False`` 인 빈 설정을 돌려줘 파이프라인이 조용히 건너뛰게 한다.

        Args:
            path: 설정 파일 경로. None 이면 프로젝트 루트의 ``config/portfolio.json``.
        """
        path = path or _DEFAULT_JSON
        if not os.path.exists(path):
            logger.info(f"{path} 없음 → 포트폴리오 백테스트 생략")
            return cls(enabled=False)

        with open(path, encoding="utf-8") as f:
            raw = _strip_comments(json.load(f))

        cfg = cls()
        cfg.enabled = bool(raw.get("enabled", True))
        cfg.name = str(raw.get("name", cfg.name))
        cfg.holdings = cls._parse_holdings(raw.get("holdings", {}))

        rb = _strip_comments(raw.get("rebalance", {}))
        cfg.rebalance = RebalanceConfig(
            enabled=bool(rb.get("enabled", True)),
            period=cls._parse_period(rb.get("period", "Q")),
            threshold=cls._parse_threshold(rb.get("threshold", 0.05)),
        )

        wt = _strip_comments(raw.get("weighting", {}))
        base_w = WeightingConfig()
        cfg.weighting = WeightingConfig(
            method=str(wt.get("method", base_w.method)).lower(),
            center=float(wt.get("center", base_w.center)),
            min_mult=float(wt.get("min", base_w.min_mult)),
            max_mult=float(wt.get("max", base_w.max_mult)),
            steepness=float(wt.get("steepness", base_w.steepness)),
        )

        if cfg.enabled:
            cfg.validate()
        return cfg

    @staticmethod
    def _parse_holdings(raw) -> Dict[str, float]:
        """holdings 를 {티커: 비중(합=1.0)} 으로 정규화한다.

        · dict{티커:비중} → 합이 1이 아니면 비율을 유지한 채 재정규화.
        · list[티커] 또는 비중이 하나라도 null → 전 종목 동일가중.
        """
        if isinstance(raw, (list, tuple)):
            items: Dict[str, Optional[float]] = {str(t): None for t in raw}
        else:
            items = {str(k): v for k, v in _strip_comments(dict(raw)).items()}
        if not items:
            return {}

        # 비중 미지정(None)이 하나라도 있으면 전체를 동일가중으로 처리
        if any(v is None for v in items.values()):
            w = 1.0 / len(items)
            return {t: w for t in items}

        total = sum(float(v) for v in items.values())
        if total <= 0:
            raise ValueError("[portfolio] holdings 비중 합이 0 이하입니다.")
        return {t: float(v) / total for t, v in items.items()}

    @staticmethod
    def _parse_period(v) -> Optional[str]:
        """주기 문자열을 정규화한다(빈값·none·off → None=주기 리밸런싱 끔)."""
        if v in (None, "", False):
            return None
        s = str(v).strip().upper()
        return None if s in ("NONE", "OFF") else s

    @staticmethod
    def _parse_threshold(v) -> Optional[float]:
        """임계값을 정규화한다(빈값·0 → None=임계 리밸런싱 끔)."""
        if v in (None, "", False) or (isinstance(v, (int, float)) and float(v) <= 0):
            return None
        return float(v)

    def validate(self) -> None:
        """설정 무결성 검사(치명은 예외, 경미한 이상은 경고)."""
        if not self.holdings:
            raise ValueError("[portfolio] holdings 가 비어 있습니다(종목:비중을 1개 이상 지정).")
        if self.rebalance.enabled and not self.rebalance.period and not self.rebalance.threshold:
            logger.warning("리밸런싱이 켜져 있으나 period·threshold 둘 다 꺼져 트리거가 없습니다. "
                           "사실상 리밸런싱하지 않습니다.")
        w = self.weighting
        if w.method not in ("none", "trend_score_sigmoid"):
            raise ValueError(f"[portfolio] 지원하지 않는 weighting.method: {w.method}")
        if w.enabled and not (0 < w.min_mult < w.max_mult):
            raise ValueError(f"[portfolio] weighting 배수는 0 < min({w.min_mult}) < max({w.max_mult}) 여야 합니다.")
