"""프로젝트 실행 설정 (JSON 로딩).

조정 가능한 값(데이터 경로·유니버스·지표 파라미터·전략 임계·비용·출력)을 프로젝트 루트의
``config.json`` 에서 읽는다. **코드 수정 없이 JSON 값만 바꿔 재실행**하면 실험 파라미터가 반영된다.

JSON 이 없거나 일부 키가 빠져도 아래 dataclass 기본값으로 폴백하므로, 부분만 적어도 동작한다.
경로 값은 프로젝트 루트 기준 상대경로로 해석한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_ROOT, "config.json")


def _abspath(p: str) -> str:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환(절대경로는 그대로)."""
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def _strip_comments(d: Dict[str, Any]) -> Dict[str, Any]:
    """JSON 에서 ``_comment`` 등 밑줄 시작 메타 키를 제거한다."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _only_known(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """dataclass 필드에 존재하는 키만 남긴다(오타·미지원 키 무시)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in _strip_comments(d).items() if k in names}


@dataclass
class TrendScoreParams:
    """일봉 TrendScore(0~100) 지표 파라미터.

    adx_penalty_max: 추세가 약할 때 최대 차감 점수.
    adx_full_strength: |ADX| 가 이 값 이상이면 페널티 0 (실질 ADX 임계값).
    """
    min_len: int = 252
    rsi_period: int = 14
    adx_period: int = 14
    ewmac_weight: float = 0.55
    tsmom_weight: float = 0.25
    rsi_weight: float = 0.20
    adx_penalty_max: float = 15.0
    adx_full_strength: float = 25.0
    smooth_span: Optional[int] = None   # EMA 스무딩 span(거래일). None=원시. 6주≈30 권장(whipsaw↓)


@dataclass
class SuperTrendParams:
    """SuperTrend(ATR 밴드 추세추종) 지표 파라미터."""
    atr_period: int = 10
    multiplier: float = 3.0


@dataclass
class Config:
    """백테스트 파이프라인 설정.

    Attributes:
        source: 시세 소스 "parquet"(로컬) | "yfinance"(온라인).
        data_dir: OHLCV parquet 디렉터리(절대경로). yfinance 소스일 땐 캐시 위치로도 쓰인다.
        start / end: 백테스트 구간("YYYY-MM-DD" 또는 None). start 이전은 워밍업 전용.
        warmup_bars: 백테스트 시작 이전에 추가 확보할 워밍업 봉 수(yfinance 다운로드 확장에 사용).
        cache: yfinance 다운로드를 data_dir 에 parquet 캐시할지 여부.
        universe: {코드: 표시명} — 백테스트 대상.
        trend_score: TrendScore 지표 파라미터.
        entry / exit: 롱-플랫 진입/청산 임계. adx_gate 없으면 entry > exit(히스테리시스),
                     있으면 entry == exit(단일 라인 크로스)도 허용.
        adx_gate: 진입 시 요구하는 최소 ADX(None=게이트 없음).
        adx_directional: True=상승방향 ADX≥gate, False=|ADX|≥gate.
        cost: 왕복 거래비용 비율(0.0010 = 0.10%).
        min_bars: 백테스트에 필요한 최소 봉 수.
        out_path: 생성할 HTML 리포트 경로(절대경로로 정규화됨).
    """
    source: str = "parquet"
    data_dir: str = os.path.join(_ROOT, "datasets", "ohlcv")
    start: Optional[str] = "2018-01-01"
    end: Optional[str] = "2026-06-30"
    warmup_bars: int = 260
    cache: bool = True
    universe: Dict[str, str] = field(default_factory=lambda: {
        "SPY": "S&P500(대형)", "QQQ": "나스닥100(성장)",
        "IWM": "러셀2000(소형)", "DIA": "다우30(우량)",
    })
    trend_score: TrendScoreParams = field(default_factory=TrendScoreParams)
    entry: float = 36.0
    exit: float = 36.0
    adx_gate: Optional[float] = 28.0
    adx_directional: bool = True
    # 비교에 포함할 전략 선택 + SuperTrend 파라미터
    use_trend_score: bool = True
    use_trend_score_base: bool = True  # 손절 없는 기본 TrendScore 변형 포함 여부(False면 +ATR 변형만)
    use_supertrend: bool = True
    use_regime_ts: bool = False  # 레짐게이트 TrendScore(진입=TS, 청산=v5.3 lifeline) 비교 포함 여부
    use_sma_slope: bool = False  # 20SMA 기울기 + ROC slope 전략 비교 포함 여부
    use_trendrider: bool = False  # 3조 regime-trendrider v4 전략 비교 포함 여부
    use_team1: bool = False  # 1조 국면·시점(ST+200EMA+%R+ATR) 전략 비교 포함 여부
    supertrend: SuperTrendParams = field(default_factory=SuperTrendParams)
    # TrendScore ATR 손절 변형(비교용). enabled 면 손절 없는 기본 + 손절 추가 변형을 함께 비교.
    stops_enabled: bool = True
    stop_atr_period: int = 14
    stop_loss_atr: Optional[float] = 2.5
    trailing_atr: Optional[float] = 3.0
    cost: float = 0.0010
    min_bars: int = 300
    out_path: str = os.path.join(_ROOT, "reports", "index.html")

    @property
    def codes(self) -> List[str]:
        return list(self.universe.keys())

    # ── 로딩 ────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """JSON 설정을 읽어 `Config` 로 만든다(누락 키는 기본값 폴백).

        Args:
            path: config.json 경로. None 이면 프로젝트 루트의 ``config.json``.
                 파일이 없으면 순수 기본값으로 동작한다.
        """
        path = path or _DEFAULT_JSON
        if not os.path.exists(path):
            print(f"[config] {path} 없음 → 내장 기본값 사용")
            return cls()

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        cfg = cls()

        # 섹션별 병합(JSON 은 섹션 구조, dataclass 는 평면 필드)
        data = _strip_comments(raw.get("data", {}))
        cfg.source = str(data.get("source", cfg.source)).lower()
        cfg.data_dir = _abspath(data.get("data_dir", "datasets/ohlcv"))
        cfg.start = data.get("start") or None
        cfg.end = data.get("end") or None
        cfg.warmup_bars = int(data.get("warmup_bars", cfg.warmup_bars))
        cfg.cache = bool(data.get("cache", cfg.cache))

        universe = _strip_comments(raw.get("universe", {}))
        if universe:
            cfg.universe = universe

        ts = _only_known(TrendScoreParams, raw.get("indicator", {}).get("trend_score", {}))
        cfg.trend_score = TrendScoreParams(**{**asdict(TrendScoreParams()), **ts})

        strat = _strip_comments(raw.get("strategy", {}))
        cfg.entry = float(strat.get("entry", cfg.entry))
        cfg.exit = float(strat.get("exit", cfg.exit))
        gate = strat.get("adx_gate", cfg.adx_gate)
        cfg.adx_gate = None if gate in (None, "", False) else float(gate)
        cfg.adx_directional = bool(strat.get("adx_directional", cfg.adx_directional))

        # 전략 선택(비교 대상) + SuperTrend 파라미터
        sel = _strip_comments(raw.get("strategies", {}))
        cfg.use_trend_score = bool(sel.get("trend_score", cfg.use_trend_score))
        cfg.use_trend_score_base = bool(sel.get("trend_score_base", cfg.use_trend_score_base))
        cfg.use_supertrend = bool(sel.get("supertrend", cfg.use_supertrend))
        cfg.use_regime_ts = bool(sel.get("regime_ts", cfg.use_regime_ts))
        cfg.use_sma_slope = bool(sel.get("sma_slope", cfg.use_sma_slope))
        cfg.use_trendrider = bool(sel.get("trendrider", cfg.use_trendrider))
        cfg.use_team1 = bool(sel.get("team1", cfg.use_team1))
        st = _only_known(SuperTrendParams, raw.get("supertrend", {}))
        cfg.supertrend = SuperTrendParams(**{**asdict(SuperTrendParams()), **st})

        stops = _strip_comments(raw.get("stops", {}))
        cfg.stops_enabled = bool(stops.get("enabled", cfg.stops_enabled))
        cfg.stop_atr_period = int(stops.get("atr_period", cfg.stop_atr_period))
        sl = stops.get("stop_loss_atr", cfg.stop_loss_atr)
        cfg.stop_loss_atr = None if sl in (None, "", False, 0) else float(sl)
        tr = stops.get("trailing_atr", cfg.trailing_atr)
        cfg.trailing_atr = None if tr in (None, "", False, 0) else float(tr)

        bt = _strip_comments(raw.get("backtest", {}))
        cfg.cost = float(bt.get("cost", cfg.cost))
        cfg.min_bars = int(bt.get("min_bars", cfg.min_bars))

        out = _strip_comments(raw.get("output", {}))
        cfg.out_path = _abspath(out.get("out_path", "reports/trendscore_swing_report.html"))

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """설정 무결성 검사(치명적 오류는 예외, 경미한 이상은 경고)."""
        if self.adx_gate is None and self.entry <= self.exit:
            raise ValueError(
                f"[config] ADX 게이트가 없으면 entry({self.entry})는 exit({self.exit})보다 커야 "
                "합니다(히스테리시스). 단일 라인 크로스를 쓰려면 adx_gate 를 설정하세요.")
        if self.entry < self.exit:
            raise ValueError(f"[config] entry({self.entry})는 exit({self.exit}) 이상이어야 합니다.")
        if self.source not in ("parquet", "yfinance"):
            raise ValueError(f"[config] source 는 'parquet'|'yfinance' 여야 합니다: {self.source}")
        if not (self.use_trend_score or self.use_supertrend or self.use_regime_ts
                or self.use_sma_slope or self.use_trendrider or self.use_team1):
            raise ValueError("[config] strategies 에서 최소 1개 전략을 활성화해야 합니다.")
        if self.warmup_bars < self.trend_score.min_len:
            print(f"[config] 경고: warmup_bars({self.warmup_bars}) < TrendScore.min_len"
                  f"({self.trend_score.min_len}). 시작 구간 지표가 예열되지 않을 수 있습니다.")
        w = self.trend_score
        wsum = w.ewmac_weight + w.tsmom_weight + w.rsi_weight
        if abs(wsum - 1.0) > 0.01:
            print(f"[config] 경고: TrendScore 가중 합이 {wsum:.2f}(권장 1.0). 그대로 진행합니다.")
