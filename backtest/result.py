"""백테스트 결과 + 성과지표 계산.

엔진이 산출한 자산곡선·거래·부가 시계열을 담고, 표준 성과지표(총수익·CAGR·Sharpe·MDD·승률 등)를
계산한다. 리포트 계층은 이 객체만 받아 렌더링하므로 지표 계산이 한 곳에 모인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .trade import Trade

_ANN = 252  # 연율화 기준 거래일 수


@dataclass
class BacktestResult:
    """단일 백테스트의 전체 산출물.

    Attributes:
        code / strategy_name: 대상 종목·전략 표시명.
        equity: 전략 자산곡선(시작 1.0 기준, DatetimeIndex).
        benchmark: Buy&Hold 자산곡선(비교용).
        trades: 왕복 거래 리스트.
        price: 원본 표준 스키마 시세(차트용).
        target_long: 봉별 목표 보유상태 bool Series(노출 계산·차트용).
        indicators: 오실레이터형 부가 시계열(예: TrendScore) — 별도 패널 차트용.
        overlays: 가격 수준형 부가 시계열(예: SuperTrend) — 가격 차트 오버레이용.
        cost: 왕복 거래비용(수수료+슬리피지) 비율.
    """
    code: str
    strategy_name: str
    equity: pd.Series
    benchmark: pd.Series
    trades: List[Trade]
    price: pd.DataFrame
    target_long: pd.Series
    indicators: Dict[str, pd.Series] = field(default_factory=dict)
    overlays: Dict[str, pd.Series] = field(default_factory=dict)
    cost: float = 0.0
    name: Optional[str] = None   # 사람이 읽는 표시명(예: "삼성전자"). None 이면 code 로 표시.
    _metrics: dict = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """화면 표시용 라벨: 표시명이 있으면 ``코드·이름``, 없으면 코드."""
        if self.name and self.name != self.code:
            return f"{self.code}·{self.name}"
        return self.code

    # ── 성과지표 ────────────────────────────────────────────────────
    @property
    def metrics(self) -> dict:
        """성과지표 딕셔너리(최초 접근 시 계산 후 캐시)."""
        if self._metrics is None:
            self._metrics = {
                "strategy": self._curve_metrics(self.equity, self.trades, self.target_long),
                "benchmark": self._curve_metrics(self.benchmark, None, None),
            }
        return self._metrics

    @classmethod
    def _curve_metrics(cls, equity: pd.Series, trades, target_long) -> dict:
        """자산곡선(+옵션 거래/노출)으로부터 표준 지표를 계산한다."""
        ret = equity.pct_change().fillna(0.0)
        years = max((equity.index[-1] - equity.index[0]).days, 1) / 365.25
        final = float(equity.iloc[-1])

        cagr = final ** (1 / years) - 1 if final > 0 else -1.0
        sharpe = float(ret.mean() / ret.std() * np.sqrt(_ANN)) if ret.std() > 0 else 0.0
        mdd = float((equity / equity.cummax() - 1.0).min())

        m = {
            "total_return_pct": (final - 1.0) * 100,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "mdd_pct": mdd * 100,
            "years": years,
        }
        if trades is not None:
            rets = np.array([t.ret for t in trades])
            m.update({
                "n_trades": len(trades),
                "win_pct": float((rets > 0).mean() * 100) if len(rets) else 0.0,
                "avg_trade_pct": float(rets.mean() * 100) if len(rets) else 0.0,
                "best_trade_pct": float(rets.max() * 100) if len(rets) else 0.0,
                "worst_trade_pct": float(rets.min() * 100) if len(rets) else 0.0,
            })
        if target_long is not None:
            m["exposure_pct"] = float(target_long.mean() * 100)
        return m

    # ── 연도별 성과 ──────────────────────────────────────────────────
    def yearly(self) -> List[dict]:
        """캘린더 연도별 성과를 계산해 리스트[dict]로 반환한다.

        각 연도 항목:
            year, strat_pct(전략 수익), bench_pct(Buy&Hold 수익), excess_pct(초과),
            mdd_pct(연중 최대낙폭), n_trades(그 해 청산 거래 수), win_pct, exposure_pct.

        수익은 일간수익 복리로 계산해 부분 연도(첫·마지막 해)도 정확히 반영한다.
        연중 MDD 는 해당 연도 구간 자체의 고점 대비로 산출(연 단위 리셋).
        """
        eq, bh = self.equity, self.benchmark
        ret_s = eq.pct_change().fillna(0.0)
        ret_b = bh.pct_change().fillna(0.0)

        rows: List[dict] = []
        for year in sorted(set(eq.index.year)):
            m = eq.index.year == year
            eq_y = eq[m]
            strat = float((1 + ret_s[m]).prod() - 1)
            bench = float((1 + ret_b[m]).prod() - 1)
            mdd = float((eq_y / eq_y.cummax() - 1.0).min())

            tr = [t for t in self.trades if t.exit_date.year == year]
            rets = np.array([t.ret for t in tr])
            exposure = float(self.target_long[self.target_long.index.year == year].mean() * 100)

            rows.append({
                "year": int(year),
                "strat_pct": strat * 100,
                "bench_pct": bench * 100,
                "excess_pct": (strat - bench) * 100,
                "mdd_pct": mdd * 100,
                "n_trades": len(tr),
                "win_pct": float((rets > 0).mean() * 100) if len(rets) else 0.0,
                "exposure_pct": exposure,
            })
        return rows

    def summary_row(self) -> dict:
        """유니버스 비교 테이블용 1행 요약(전략 지표 + 종목 정보)."""
        s = self.metrics["strategy"]
        b = self.metrics["benchmark"]
        return {
            "code": self.code,
            "total_return_pct": s["total_return_pct"],
            "cagr_pct": s["cagr_pct"],
            "sharpe": s["sharpe"],
            "mdd_pct": s["mdd_pct"],
            "n_trades": s.get("n_trades", 0),
            "win_pct": s.get("win_pct", 0.0),
            "exposure_pct": s.get("exposure_pct", 0.0),
            "bh_cagr_pct": b["cagr_pct"],
            "bh_mdd_pct": b["mdd_pct"],
        }
