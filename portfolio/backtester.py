"""자산배분 포트폴리오 백테스터.

여러 종목을 목표비중으로 동시 보유하며 **주기(달력) + 임계(드리프트)** 리밸런싱을 시뮬레이션한다.
단일종목 롱-플랫 `Backtester` 와 달리 복수 자산의 일간수익을 비중 합성해 자산곡선을 만든다.

체결 규약: 일간 종가 기준으로 평가·리밸런싱한다(결정과 실행이 같은 종가에서 이뤄지므로 룩어헤드 없음).
비교 벤치마크는 **동일 초기비중을 리밸런싱 없이 보유(드리프트)** 한 곡선으로, 리밸런싱의 순효과가
자산곡선 대 벤치마크 차이로 드러난다.

산출물은 기존 `BacktestResult` 로 포장해 리포트·성과지표 계층을 그대로 재사용한다(관심사 분리).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest import BacktestResult, Trade
from data import DataLoader

from .config import PortfolioConfig, RebalanceConfig

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """비중 기반 자산배분 + 리밸런싱 백테스터.

    Args (생성자):
        loader: 종목 시세를 표준 스키마로 읽는 `DataLoader`(parquet·yfinance 등).
        cost: 왕복 거래비용 비율(예 0.0010 = 0.10%). 리밸런싱 회전율에 비례해 차감.
    """

    def __init__(self, loader: DataLoader, cost: float = 0.0010):
        self.loader = loader
        self.cost = cost

    # ── public ──────────────────────────────────────────────────────
    def run(self, pcfg: PortfolioConfig, start=None, end=None) -> BacktestResult:
        """포트폴리오를 백테스트해 `BacktestResult` 로 반환한다.

        Args:
            pcfg: 포트폴리오 설정(종목·비중·리밸런싱).
            start / end: 백테스트 구간("YYYY-MM-DD"·Timestamp·None). None 이면 미제한.
        Returns:
            자산곡선·벤치마크·성과지표를 담은 `BacktestResult`(code="PORTFOLIO").
        """
        prices, weights = self._load_prices(pcfg.holdings)
        closes = self._align_closes(prices, start, end)

        equity, rb_dates = self._simulate(closes, weights, pcfg.rebalance)
        benchmark = self._buy_and_hold(closes, weights)

        logger.info(f"포트폴리오 시뮬레이션 · {len(closes.columns)}종목 · "
                    f"{closes.index[0]:%Y-%m-%d}~{closes.index[-1]:%Y-%m-%d} · "
                    f"리밸런싱 {len(rb_dates)}회")
        return self._to_result(pcfg, closes, weights, equity, benchmark, rb_dates)

    # ── 데이터 준비 ─────────────────────────────────────────────────
    def _load_prices(self, holdings: Dict[str, float]) -> Tuple[Dict, Dict[str, float]]:
        """보유 종목 시세를 로드한다. 실패 종목은 제외하고 남은 비중을 재정규화한다."""
        prices = {}
        for code in holdings:
            try:
                prices[code] = self.loader.load(code)
            except Exception as exc:  # noqa: BLE001 — 개별 종목 실패가 전체를 막지 않도록
                logger.warning(f"{code}: 로드 실패 → 포트폴리오에서 제외 ({exc})")
        if not prices:
            raise RuntimeError("포트폴리오: 로드 가능한 종목이 없습니다.")

        kept = {c: holdings[c] for c in prices}
        total = sum(kept.values())
        weights = {c: w / total for c, w in kept.items()}
        dropped = [c for c in holdings if c not in prices]
        if dropped:
            logger.warning(f"제외된 종목 {dropped} → 남은 {len(weights)}종목으로 비중 재정규화")
        return prices, weights

    @staticmethod
    def _align_closes(prices: Dict, start, end) -> pd.DataFrame:
        """종목별 종가를 하나의 DataFrame 으로 정렬한다(공통 거래일만).

        어떤 종목이든 값이 없는 날은 버려(dropna) 모든 종목이 거래된 날짜로 구간을 맞춘다.
        상장일이 다른 종목을 섞으면 가장 늦은 상장일 이후부터 백테스트된다.
        """
        closes = pd.DataFrame({c: p.df["close"] for c, p in prices.items()}).sort_index()
        closes = closes.loc[start:end].dropna()
        if len(closes) < 2:
            raise ValueError("포트폴리오: 공통 거래일이 부족합니다(종목 구간이 겹치지 않음).")
        return closes

    # ── 시뮬레이션 ──────────────────────────────────────────────────
    def _simulate(self, closes: pd.DataFrame, weights: Dict[str, float],
                  rb: RebalanceConfig) -> Tuple[pd.Series, List[pd.Timestamp]]:
        """비중 합성 자산곡선 + 리밸런싱 시뮬레이션.

        각 종목 가치를 일간수익으로 굴리고, 주기·임계 트리거가 걸리면 목표비중으로 되돌린다.
        리밸런싱 시 회전율(단방향)에 비례해 왕복비용을 차감한다.

        Returns:
            (equity: 시작 1.0 자산곡선, rb_dates: 실제 리밸런싱이 일어난 날짜 리스트).
        """
        tickers = list(closes.columns)
        w_target = np.array([weights[t] for t in tickers], dtype=float)
        rets = closes.pct_change().fillna(0.0).to_numpy()  # 일간 종목수익 행렬
        dates = closes.index
        n = len(dates)

        # 주기(달력) 리밸런싱 날짜 마스크(리밸런싱 꺼져 있으면 전부 False)
        periodic = (self._periodic_mask(dates, rb.period)
                    if rb.enabled else np.zeros(n, dtype=bool))
        thr = rb.threshold if rb.enabled else None

        value = w_target.copy()          # 자산별 가치(첫날 목표비중, 합=1.0)
        eq = np.empty(n)
        rb_dates: List[pd.Timestamp] = []

        for i in range(n):
            if i > 0:
                value = value * (1.0 + rets[i])   # 당일 수익 반영
            total = value.sum()
            w_now = value / total                 # 현재 실제 비중

            # 리밸런싱 판단(첫날은 이미 목표비중이라 제외)
            if i > 0 and rb.enabled:
                drift = np.abs(w_now - w_target)
                trigger = periodic[i] or (thr is not None and drift.max() > thr)
                if trigger:
                    # 회전율(단방향) = 목표로 되돌리며 사고파는 비중의 절반.
                    # 왕복비용을 회전 비중에 비례해 차감(간이 마찰 모델).
                    turnover = 0.5 * drift.sum()
                    total *= (1.0 - self.cost * turnover)
                    value = w_target * total       # 목표비중 복원
                    rb_dates.append(dates[i])
            eq[i] = total

        return pd.Series(eq, index=dates, name="equity"), rb_dates

    @staticmethod
    def _periodic_mask(dates: pd.DatetimeIndex, period: str | None) -> np.ndarray:
        """주기 리밸런싱이 일어나는 날(각 주기의 첫 거래일)을 True 로 표시한다.

        period: "M"·"Q"·"Y"·"W" 는 해당 달력 단위가 바뀌는 첫 거래일, "<N>D" 는 N거래일마다.
        None/미지원 값이면 전부 False(주기 리밸런싱 없음).
        """
        n = len(dates)
        mask = np.zeros(n, dtype=bool)
        if not period:
            return mask

        # "<N>D": N거래일 간격
        if period.endswith("D") and period[:-1].isdigit():
            step = int(period[:-1])
            if step > 0:
                mask[step::step] = True
            return mask

        # 달력 경계(월/분기/연/주)로 그룹키를 만들고, 키가 바뀌는 첫 거래일을 표시
        key_fn = {
            "M": lambda d: (d.year, d.month),
            "Q": lambda d: (d.year, (d.month - 1) // 3),
            "Y": lambda d: (d.year,),
            "A": lambda d: (d.year,),
            "W": lambda d: d.isocalendar()[:2],  # (ISO 연도, ISO 주차)
        }.get(period)
        if key_fn is None:
            logger.warning(f"알 수 없는 리밸런싱 주기 '{period}' → 주기 리밸런싱 없이 진행")
            return mask

        prev = None
        for i, d in enumerate(dates):
            k = key_fn(d)
            if prev is not None and k != prev:
                mask[i] = True
            prev = k
        return mask

    @staticmethod
    def _buy_and_hold(closes: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """벤치마크: 초기 비중으로 사서 **리밸런싱 없이 보유**한 자산곡선(시작 1.0)."""
        norm = closes / closes.iloc[0]                       # 종목별 정규화(시작 1.0)
        w = np.array([weights[t] for t in closes.columns])
        bh = (norm.to_numpy() * w).sum(axis=1)
        return pd.Series(bh, index=closes.index, name="buy_and_hold")

    @staticmethod
    def _rebalance_trades(equity: pd.Series, rb_dates: List[pd.Timestamp]) -> List[Trade]:
        """리밸런싱 구간을 각각 하나의 보유거래로 기록한다(리포트 거래 테이블·활동 집계용).

        연속한 리밸런싱 시점(과 시작/종료)을 경계로 자산곡선을 구간 분할하고, 각 구간을
        entry~exit 로 보는 `Trade` 를 만든다. ret 은 그 구간의 포트폴리오 수익률(리밸런싱 비용
        반영 후). 마지막 구간은 청산 사유 'eod', 나머지는 '리밸런싱'.
        """
        idx = equity.index
        # 경계 = 시작 + (양끝과 겹치지 않는) 리밸런싱일 + 종료
        bounds = [idx[0]] + [d for d in rb_dates if d not in (idx[0], idx[-1])] + [idx[-1]]
        trades: List[Trade] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            pa, pb = idx.get_loc(a), idx.get_loc(b)
            eq_a, eq_b = float(equity.iloc[pa]), float(equity.iloc[pb])
            trades.append(Trade(
                entry_date=a, exit_date=b, entry_px=eq_a, exit_px=eq_b,
                ret=eq_b / eq_a - 1.0, bars_held=pb - pa,
                exit_reason=("eod" if b == idx[-1] else "리밸런싱")))
        return trades

    # ── 결과 포장 ───────────────────────────────────────────────────
    def _to_result(self, pcfg: PortfolioConfig, closes: pd.DataFrame,
                   weights: Dict[str, float], equity: pd.Series,
                   benchmark: pd.Series, rb_dates: List[pd.Timestamp]) -> BacktestResult:
        """자산곡선을 기존 `BacktestResult` 로 감싼다(리포트·지표 재사용).

        · price: 가격 패널에 '리밸런싱 없는 바스켓(벤치마크)' 곡선을 실어 대비를 보여준다.
        · target_long: 포트폴리오는 항상 전액 투자이므로 전 구간 True(노출 100%).
        · trades: 리밸런싱 구간별 보유거래(개별 왕복거래 대신 리밸런싱 활동을 표현).
        """
        bench_vals = benchmark.to_numpy()
        price_df = pd.DataFrame(
            {"open": bench_vals, "high": bench_vals, "low": bench_vals, "close": bench_vals},
            index=closes.index)
        target_long = pd.Series(True, index=closes.index)

        return BacktestResult(
            code="PORTFOLIO",
            strategy_name=f"{len(weights)}종목 {pcfg.rebalance.describe()}",
            name=pcfg.name,
            equity=equity,
            benchmark=benchmark,
            trades=self._rebalance_trades(equity, rb_dates),
            price=price_df,
            target_long=target_long,
            indicators={},
            overlays={},
            cost=self.cost,
        )
