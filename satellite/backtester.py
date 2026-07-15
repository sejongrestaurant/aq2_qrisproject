"""사테라이트(모멘텀 로테이션) 백테스터.

후보 유니버스의 각 종목에 지표(TrendScore)를 계산해 매 체크주기마다 **점수 상위 top_n 슬롯을
동일가중으로 보유**하고, 다음 체크에서 상위 구성이 바뀌면 교체한다. 룩어헤드를 막기 위해 종가
기준 점수로 선정한 목표를 **다음 거래일에 반영**하고, 종목 교체 시 회전율에 비례해 비용을 뺀다.

두 가지 위험관리를 얹는다:
  · **트레일링 스탑** — 보유 종목이 진입 후 고점 대비 밀리면(ATR 다이나믹 또는 고정 %) 청산.
  · **현금 대피(BIL)** — 자격 종목이 top_n 미만이거나 손절된 슬롯은 현금 대용(BIL)으로 보유.

상장 시점이 다른 종목이 섞여도, 각 종목은 가격·점수가 유효한 날에만 선정 후보가 된다(상장 전·
지표 워밍업 구간은 자동 제외). 산출물은 기존 `BacktestResult` 로 포장해 리포트를 재사용한다.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import BacktestResult
from data import DataLoader
from indicator import Indicator
from portfolio.schedule import period_mask, segment_trades

from .config import SatelliteConfig
from .trailing import TrailingStop

logger = logging.getLogger(__name__)


class SatelliteBacktester:
    """지표 점수 상위 top_n 동일가중 로테이션 백테스터(트레일링 스탑 + BIL 현금).

    Args (생성자):
        loader: 종목 시세를 표준 스키마로 읽는 `DataLoader`.
        indicator: 순위 산정용 지표(예: `TrendScoreIndicator`). 각 종목 종가 시계열에 적용.
        cost: 왕복 거래비용 비율(예 0.0010). 종목 교체 회전율에 비례해 차감.
    """

    def __init__(self, loader: DataLoader, indicator: Indicator, cost: float = 0.0010):
        self.loader = loader
        self.indicator = indicator
        self.cost = cost

    # ── public ──────────────────────────────────────────────────────
    def run(self, scfg: SatelliteConfig, start=None, end=None,
            trailing: Optional[TrailingStop] = None) -> BacktestResult:
        """사테라이트 로테이션을 백테스트해 `BacktestResult` 로 반환한다.

        Args:
            scfg: 사테라이트 설정(유니버스·top_n·체크주기·현금 대용).
            start / end: 백테스트 구간(None 이면 미제한).
            trailing: 트레일링 스탑 규칙(None 이면 스탑 없이 만기 보유). ATR 방식이면
                      해당 기간으로 ATR 을 계산해 넘긴다.
        Returns:
            자산곡선·벤치마크를 담은 `BacktestResult`(code="SATELLITE").
        """
        atr_period = trailing.atr_period if (trailing and trailing.needs_atr) else None
        closes, scores, atr, cash_ret = self._load_matrix(
            scfg.universe, scfg.cash_ticker, atr_period, start, end)
        equity, rotations, stops, last_pick, pick_log = self._simulate(
            closes, scores, atr, cash_ret, scfg.top_n, scfg.check_period, trailing,
            scfg.entry_score, scfg.exit_score)
        benchmark = self._equal_weight_all(closes)

        rule = trailing.name if trailing else "스탑없음"
        logger.info(f"사테라이트[{rule}] 시뮬레이션 · 후보 {closes.shape[1]}종목 · "
                    f"{closes.index[0]:%Y-%m-%d}~{closes.index[-1]:%Y-%m-%d} · "
                    f"교체 {len(rotations)}회 · 손절 {stops}회 · 최근 보유 {last_pick}")
        return self._to_result(scfg, closes, equity, benchmark, rotations, trailing, pick_log)

    # ── 데이터 준비 ─────────────────────────────────────────────────
    def _load_matrix(self, universe: List[str], cash_ticker: str,
                     atr_period: Optional[int], start, end
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.Series]:
        """유니버스의 종가·지표점수(·ATR) 행렬과 현금 대용 일간수익을 만든다.

        날짜는 모든 후보의 합집합으로 두고(상장 전은 NaN), 지표·ATR 은 각 종목에 계산 후 같은
        날짜축에 정렬한다. 이렇게 하면 상장이 늦은 종목도 유효해지는 날부터 선정 후보가 된다.
        ATR 은 트레일링 스탑이 ATR 다이나믹일 때만 계산한다(고정 방식이면 None).
        """
        close_cols: Dict[str, pd.Series] = {}
        score_cols: Dict[str, pd.Series] = {}
        atr_cols: Dict[str, pd.Series] = {}
        for code in universe:
            try:
                price = self.loader.load(code)
            except Exception as exc:  # noqa: BLE001 — 개별 종목 실패가 전체를 막지 않도록
                logger.warning(f"{code}: 로드 실패 → 후보에서 제외 ({exc})")
                continue
            df = price.df
            close_cols[code] = df["close"]
            score_cols[code] = self.indicator.compute(df)
            if atr_period is not None:
                atr_cols[code] = self._atr(df["high"], df["low"], df["close"], atr_period)
        if not close_cols:
            raise RuntimeError("사테라이트: 로드 가능한 후보 종목이 없습니다.")

        closes = pd.DataFrame(close_cols).sort_index().loc[start:end]
        # 모든 후보가 아직 없던 초기 구간(전 종목 NaN 행)은 버린다.
        closes = closes.dropna(how="all")
        if len(closes) < 2:
            raise ValueError("사테라이트: 백테스트할 거래일이 부족합니다.")
        scores = pd.DataFrame(score_cols).reindex(index=closes.index, columns=closes.columns)
        atr = (pd.DataFrame(atr_cols).reindex(index=closes.index, columns=closes.columns)
               if atr_period is not None else None)
        cash_ret = self._load_cash_returns(cash_ticker, closes.index)
        return closes, scores, atr, cash_ret

    def _load_cash_returns(self, cash_ticker: str, index: pd.DatetimeIndex) -> pd.Series:
        """현금 대용(BIL) 일간수익을 백테스트 날짜축에 맞춰 만든다.

        로드 실패(오프라인·파일 없음) 시에는 **무이자 현금**(수익 0)으로 대체하고 경고만 남겨,
        현금 대용 데이터가 없어도 파이프라인이 멈추지 않게 한다.
        """
        try:
            price = self.loader.load(cash_ticker)
        except Exception as exc:  # noqa: BLE001 — 현금 대용 없으면 무이자 현금 폴백
            logger.warning(f"{cash_ticker}: 현금 대용 로드 실패 → 무이자 현금(수익 0)으로 대체 ({exc})")
            return pd.Series(0.0, index=index)
        close = price.df["close"].reindex(index).ffill()
        return close.pct_change(fill_method=None).fillna(0.0)

    # ── 시뮬레이션 ──────────────────────────────────────────────────
    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame,
                  atr: Optional[pd.DataFrame], cash_ret: pd.Series, top_n: int,
                  period: str, trailing: Optional[TrailingStop],
                  entry_score: float, exit_score: float
                  ) -> Tuple[pd.Series, List[pd.Timestamp], int, List[str],
                             List[Tuple[pd.Timestamp, List[str]]]]:
        """점수 게이트 로테이션(상위 top_n) + 트레일링 스탑 + BIL 현금 시뮬레이션.

        각 체크일 종가 점수로 **자격 종목**을 추린 뒤 상위 슬롯을 뽑아 **다음 거래일**에
        목표비중(1/top_n)으로 재조정한다. 자격 판정은 히스테리시스를 둔다:
          · 미보유 종목 → 점수 ≥ entry_score 여야 신규 편입.
          · 보유 종목   → 점수 ≥ exit_score 면 유지, 미만이면 청산.
        (entry_score > exit_score 이므로 그 사이 구간은 '보유만 유지'해 잦은 교체를 억제.)
        자격 종목이 top_n 미만이면 빈 슬롯은 현금 대용(BIL)으로 채운다(무조건 채우지 않음).
        보유 중에는 매일 트레일링 스탑을 점검해, 손절선 이하로 내려온 종목은 현금(BIL)으로 대피시킨다.

        룩어헤드 방지: 손절은 당일 종가로 '판정' 하고 대피는 그날 종가(당일 슬롯 무효화)에서
        반영해 다음날부터 현금 수익을 받는다. 재조정은 전 체크일 결정을 오늘 반영한다.

        보유 슬롯의 고점(peak)은 재조정 때마다 그날 종가로 재설정한다(매 주기 '새 포지션' 으로
        간주 — 지난 주기 고점이 이월돼 스탑이 과하게 느슨해지는 것을 막는다).

        Returns:
            (equity 시작 1.0, rotations 교체일 리스트, stops 손절 횟수, last_pick 마지막 보유 종목,
             pick_log 보유구성이 바뀐 시점별 (날짜, 선정 티커 리스트)).
        """
        idx = closes.index
        tickers = list(closes.columns)
        C = closes.to_numpy(dtype=float)
        R = np.zeros_like(C)
        R[1:] = C[1:] / C[:-1] - 1.0                 # 일간수익(가격 결측 구간은 NaN)
        S = scores.to_numpy(dtype=float)             # 점수(워밍업·상장전 NaN)
        A = atr.to_numpy(dtype=float) if atr is not None else None
        cash_r = cash_ret.to_numpy(dtype=float)
        T, N = C.shape
        check = period_mask(idx, period)

        name_val = np.zeros(N)   # 종목별 보유 가치
        cash_val = 0.0           # 현금 대용(BIL) 슬롯 가치
        peak = np.full(N, np.nan)  # 보유구간 종가 고점(미보유는 NaN)
        holding = False
        pending: Optional[Tuple[np.ndarray, float]] = None  # (다음날 종목비중, 현금비중)
        eq = np.empty(T)
        rotations: List[pd.Timestamp] = []
        pick_log: List[Tuple[pd.Timestamp, List[str]]] = []
        stops = 0
        prev_set: frozenset = frozenset()

        for i in range(T):
            # (a) 당일 수익 반영(첫 투자 이후). 종목 결측 수익은 0, 현금은 BIL 수익.
            if holding:
                name_val = name_val * (1.0 + np.nan_to_num(R[i], nan=0.0))
                cash_val = cash_val * (1.0 + cash_r[i])
            # (b) 전 체크에서 정한 목표를 오늘 반영(교체/재조정 + 회전율 비용)
            if pending is not None:
                w_names, w_cash = pending
                total = (name_val.sum() + cash_val) if holding else 1.0
                if holding and total > 0:
                    w_now = np.concatenate([name_val / total, [cash_val / total]])
                else:
                    w_now = np.zeros(N + 1)
                w_tgt = np.concatenate([w_names, [w_cash]])
                turnover = 0.5 * np.abs(w_tgt - w_now).sum()
                total *= (1.0 - self.cost * turnover)
                name_val = w_names * total
                cash_val = w_cash * total
                peak = np.where(w_names > 0, C[i], np.nan)  # 새 보유 슬롯 고점=오늘 종가
                holding = True
                pending = None
            eq[i] = (name_val.sum() + cash_val) if holding else 1.0
            # (c) 트레일링 스탑 점검(당일 종가) → 손절 슬롯을 현금(BIL)으로 대피
            if holding and trailing is not None:
                held = np.where(name_val > 0.0)[0]
                for j in held:
                    if np.isnan(C[i, j]):
                        continue
                    peak[j] = C[i, j] if np.isnan(peak[j]) else max(peak[j], C[i, j])
                    atr_j = A[i, j] if A is not None else np.nan
                    if C[i, j] <= trailing.stop_level(peak[j], atr_j):
                        cash_val += name_val[j]      # 손절 자금 → 현금 대용
                        name_val[j] = 0.0
                        peak[j] = np.nan
                        stops += 1
            # (d) 오늘 종가 점수로 자격 종목 선정(히스테리시스) → 다음 거래일 목표로 예약
            if check[i]:
                row = S[i]
                # 보유 여부에 따라 다른 문턱: 미보유=entry, 보유=exit(현재 실제 보유 슬롯 기준).
                held_now = name_val > 0.0
                thr = np.where(held_now, exit_score, entry_score)
                elig = np.where(~np.isnan(row) & ~np.isnan(C[i]) & (row >= thr))[0]
                ranked = elig[np.argsort(row[elig])[::-1]] if len(elig) else elig  # 점수 내림차순
                sel = ranked[:top_n]
                w_names = np.zeros(N)
                if len(sel):
                    w_names[sel] = 1.0 / top_n                        # 슬롯당 1/top_n
                w_cash = 1.0 - len(sel) / top_n                       # 빈 슬롯 → 현금(BIL)
                pending = (w_names, w_cash)                           # 자격 0개면 전액 현금
                cur_set = frozenset(int(s) for s in sel)
                if cur_set != prev_set:                               # 보유 구성 변경 = 교체
                    rotations.append(idx[i])
                    # 선정 종목을 점수 내림차순으로 기록(로테이션 내역 리포트용).
                    pick_log.append((idx[i], [tickers[j] for j in sel]))
                    prev_set = cur_set

        last_pick = [tickers[j] for j in sorted(prev_set)] if prev_set else []
        return pd.Series(eq, index=idx, name="equity"), rotations, stops, last_pick, pick_log

    @staticmethod
    def _equal_weight_all(closes: pd.DataFrame) -> pd.Series:
        """벤치마크: 후보 전 종목을 매일 동일가중 보유(선정·스탑 없이 '전부 보유')한 자산곡선."""
        daily = closes.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
        return (1.0 + daily).cumprod().rename("buy_and_hold")

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Wilder ATR(참 범위의 지수이동평균). SuperTrend 지표와 동일 정의."""
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    # ── 결과 포장 ───────────────────────────────────────────────────
    def _to_result(self, scfg: SatelliteConfig, closes: pd.DataFrame, equity: pd.Series,
                   benchmark: pd.Series, rotations: List[pd.Timestamp],
                   trailing: Optional[TrailingStop],
                   pick_log: List[Tuple[pd.Timestamp, List[str]]]) -> BacktestResult:
        """자산곡선을 기존 `BacktestResult` 로 감싼다(리포트·지표 재사용).

        · price: 가격 패널에 '전 종목 동일가중(벤치마크)' 곡선을 실어 선정 효과를 대비.
        · target_long: 항상 전액 투자(종목 또는 현금 대용)이므로 전 구간 True.
        · trades: 종목 교체 구간별 보유거래(교체 활동을 리포트 거래 테이블로 표현).
        · rotations_log: 매 교체 시점의 선정 종목·구간수익(리포트 로테이션 내역 표).
        · strategy_name: 트레일링 방식을 붙여 두 변형을 헤드투헤드로 구분한다.
        """
        bench_vals = benchmark.to_numpy()
        price_df = pd.DataFrame(
            {"open": bench_vals, "high": bench_vals, "low": bench_vals, "close": bench_vals},
            index=closes.index)
        target_long = pd.Series(True, index=closes.index)

        rule = trailing.name if trailing else "스탑없음"
        return BacktestResult(
            code="SATELLITE",
            strategy_name=f"Top{scfg.top_n} {scfg.check_period}체크 · {rule}",
            name=scfg.name,
            equity=equity,
            benchmark=benchmark,
            trades=segment_trades(equity, rotations, reason="교체"),
            price=price_df,
            target_long=target_long,
            indicators={},
            overlays={},
            cost=self.cost,
            rotations_log=self._build_rotations_log(equity, pick_log, scfg),
        )

    @staticmethod
    def _build_rotations_log(equity: pd.Series, pick_log: List[Tuple[pd.Timestamp, List[str]]],
                             scfg: SatelliteConfig) -> List[dict]:
        """교체 시점별 선정 종목·구간수익을 리포트용 dict 리스트로 만든다.

        구간수익은 이 교체일부터 다음 교체일(마지막은 종료일)까지의 자산곡선 변화로 계산한다.
        종목명은 설정의 names 맵으로 사람이 읽는 라벨(``코드·이름``)로 바꾼다.
        """
        idx = equity.index
        out: List[dict] = []
        for k, (date, codes) in enumerate(pick_log):
            nxt = pick_log[k + 1][0] if k + 1 < len(pick_log) else idx[-1]
            pa, pb = idx.get_loc(date), idx.get_loc(nxt)
            ret = float(equity.iloc[pb] / equity.iloc[pa] - 1.0) if pb > pa else 0.0
            out.append({
                "date": date,
                "labels": [scfg.label(c) for c in codes],
                "n": len(codes),
                "ret_pct": ret * 100.0,
            })
        return out
