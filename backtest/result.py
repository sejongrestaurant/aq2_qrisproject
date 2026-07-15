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
        rotations_log: 모멘텀 로테이션 선정 이력(선택). 각 원소는 보유구성이 바뀐 시점의
            {"date","labels"(선정 종목 표시명 리스트),"n","ret_pct"(다음 교체까지 구간수익)}.
            사테라이트·IRP 처럼 '그때그때 어떤 종목을 골랐는지' 를 리포트 표로 보여주기 위한 것.
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
    rotations_log: Optional[List[dict]] = None
    benchmark_name: str = "Buy&Hold"  # 벤치마크 곡선 표시명(예: "KODEX TRF7030"). 리포트 라벨용.
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

        # 하방 위험·낙폭 스트레스 지표(공격적 전략 평가에 Sharpe 보완):
        #  · Sortino  = 상승 변동성은 빼고 하락 변동성만으로 평가(MAR=0 기준 하방편차 연율화).
        #  · Calmar   = CAGR / |MDD|. 최악 낙폭 대비 수익 효율.
        #  · Ulcer    = 낙폭(%) 시계열의 RMS. 낙폭의 깊이·지속을 함께 반영(체감 스트레스).
        downside = np.minimum(ret.to_numpy(), 0.0)                 # 0 미만 일간수익만(MAR=0)
        dd_dev = float(np.sqrt(np.mean(downside ** 2)))            # 하방편차(일간)
        sortino = float(ret.mean() / dd_dev * np.sqrt(_ANN)) if dd_dev > 0 else 0.0
        calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
        dd_curve = (equity / equity.cummax() - 1.0).to_numpy() * 100.0  # 낙폭(%) 시계열(≤0)
        ulcer = float(np.sqrt(np.mean(dd_curve ** 2)))

        # 회복(언더워터) 분석: 고점에서 밀려 새 고점을 회복하기까지의 기간(달력일).
        #  · 평균/최장 회복일수 → 낙폭이 얼마나 오래 지속되는지(체감 스트레스의 '지속' 축).
        #  · 연평균 신규 고점 갱신 → 새 고점을 얼마나 자주 찍는지(전진 빈도).
        avg_rec, max_rec, new_highs_py = cls._recovery_stats(equity, years)

        m = {
            "total_return_pct": (final - 1.0) * 100,
            "cagr_pct": cagr * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "ulcer": ulcer,
            "mdd_pct": mdd * 100,
            "avg_recovery_days": avg_rec,
            "max_recovery_days": max_rec,
            "new_highs_per_year": new_highs_py,
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

    @staticmethod
    def _recovery_stats(equity: pd.Series, years: float):
        """언더워터(고점→회복) 구간 분석.

        고점에서 밀려 이전 고점을 회복(=새 고점 도달)하기까지 걸린 달력일수를 구간마다 모아
        평균·최장을 계산하고, 새 고점을 찍은 날 수를 연평균으로 환산한다. 마지막까지 회복하지
        못한(아직 언더워터) 구간은 완결되지 않았으므로 평균·최장에서 제외한다.

        Returns:
            (avg_recovery_days, max_recovery_days, new_highs_per_year).
        """
        cmx = equity.cummax()
        under = (equity < cmx).to_numpy()
        dates = equity.index
        new_high_days = int((cmx.diff().fillna(0.0).to_numpy() > 0).sum())  # cummax 가 오른 날

        recoveries: List[int] = []
        in_dd = False
        peak_date = dates[0]              # 직전 고점 날짜(하락 시작 기준점)
        for k in range(len(under)):
            if under[k]:
                in_dd = True              # 고점 아래 = 언더워터
            else:
                if in_dd:                 # 고점 회복 → 구간 종료(peak→회복 달력일)
                    recoveries.append((dates[k] - peak_date).days)
                    in_dd = False
                peak_date = dates[k]      # 새/동일 고점 갱신 → 기준점 이동
        avg_rec = float(np.mean(recoveries)) if recoveries else 0.0
        max_rec = float(np.max(recoveries)) if recoveries else 0.0
        new_highs_py = new_high_days / years if years > 0 else 0.0
        return avg_rec, max_rec, new_highs_py

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
