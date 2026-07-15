"""사테라이트 백테스터 V2 — '부분 진입(점수 비례 슬롯 경사)' 실험용 변형.

원본(`SatelliteBacktester`)은 수정하지 않는다. 원본의 슬롯 판정은 **이진**이다:
TrendScore ≥ entry_score(60)면 슬롯을 가득(1/top_n) 채우고 미달이면 0 — 59점과 61점의
취급이 극단적으로 갈린다. 진단된 약점(전환 구간 5슬롯대 평균 구간수익 −1.08%, 유일한
마이너스 상태)이 바로 이 문턱 근처에서 나온다: 급락 후 회복 국면에서 점수가 문턱을
오르내리는 동안 노출이 0과 100%를 널뛴다.

변형 가설: 슬롯 충전율을 점수 비례 경사로 바꾸면(예 50점 40% → 60점 100%) 전환 구간에서
노출이 매끄럽게 복원돼 V자 반등 재진입 지연이 완화된다. 대신 문턱 아래 종목에 미리
노출되므로 베어랠리 오탐 시 손실이 커질 수 있다(트레이드오프).

설계 결정 — 경사는 **신규·보유 양쪽에 같은 규칙으로** 적용한다:
    슬롯 충전율 = f(점수)  (진입이든 유지든 동일)
보유 종목만 예외로 가득 채우면 같은 점수에서 크기가 경로에 따라 달라져(신규 55점=70%,
보유 55점=100%) 규칙이 자기모순이 되고, 그 불일치가 곧 교체 회전으로 새 나간다.
히스테리시스는 충전율이 아니라 **자격 문턱**으로 유지한다(신규는 ramp_score 이상,
보유는 exit_score 이상까지 하한 충전율로 잔류).

슬롯 **수**는 여전히 top_n 고정이다(슬롯 수 = 국면 대응이라는 원 설계를 건드리지 않는다).
바뀌는 것은 '슬롯을 얼마나 채우는가' 뿐이고, 못 채운 몫은 원본과 같이 현금 대용(단기채권)
으로 간다 — 즉 경사 진입은 실효 채권비중을 연속적으로 조절하는 것과 같다.

축퇴(degenerate) 보장: `ramp_score=None` 이면 f ≡ 1.0 이 되어 원본과 **부동소수점까지
동일**한 결과를 낸다(회귀 검증용 기준선).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.schedule import period_mask

from .backtester import SatelliteBacktester
from .trailing import TrailingStop

logger = logging.getLogger(__name__)


class SatelliteBacktesterV2(SatelliteBacktester):
    """슬롯 충전율을 TrendScore 비례 경사로 채우는 사테라이트 변형.

    Args (생성자):
        loader / indicator / cost: 원본과 동일(그대로 위임).
        ramp_score: 슬롯 경사의 **아래쪽 끝** 점수(여기서 ramp_floor 만큼 채움). None 이면
            원본 이진 게이트로 축퇴한다(= entry_score 에서 100%). 예 50.0.
        full_score: 슬롯을 100% 채우는 점수. None 이면 설정의 entry_score(60)를 쓴다.
            → `ramp_score=50` 만 주면 '50점 하한충전 → 60점 만충' 경사가 된다.
        entry_gate: **신규 편입 자격 문턱**(랭킹 진입 자격). None 이면 ramp_score 를 따라간다.
            크기(경사)와 자격(문턱)은 별개다 — 이 둘을 묶어 두면 실험이 오염된다.
            60점 문턱의 실제 역할은 품질 게이트라기보다 **기득권 보호**다: 45↔60 의 넓은
            히스테리시스 덕에 보유 종목은 60 이상의 '압도적으로 나은' 신규에게만 밀린다.
            문턱을 50으로 낮추면 보호 폭이 45↔50 으로 좁아져, 일시 조정으로 45~50 에 떨어진
            주도주가 50~60 짜리 그저 그런 신규에게 축출된다(2026-04 실측: 이 축출로 한 구간
            +22.8% → +7.9%). 그래서 크기만 조절하고 싶으면 entry_gate 는 60 으로 고정한다.
        ramp_floor: ramp_score 에서의 슬롯 충전율(0~1]. 예 0.4 = 40%만 채움.
            보유 종목이 ramp_score 아래(exit_score 까지)로 밀린 구간도 이 하한을 쓴다.
        ramp_hold: 경사를 **보유 종목에도** 적용할지.
            True  = 신규·보유 동일 규칙(점수만으로 크기 결정). 경로 무관하지만 '부분 진입'과
                    '부분 유지(축소)' 두 메커니즘이 동시에 켜진다.
            False = **부분 진입만**. 보유 종목은 exit_score 위면 슬롯을 가득 유지한다
                    → 시험 진입(첫 달 부분) 후 살아남으면 다음 체크에서 만충되는 단계적 진입.
                    같은 점수라도 신규/보유에 따라 크기가 달라진다(경로 의존).
            두 메커니즘을 분리 측정하기 위한 손잡이다(규율: 한 번에 하나만 변경).
    """

    def __init__(self, loader, indicator, cost: float = 0.0010, *,
                 ramp_score: Optional[float] = None,
                 full_score: Optional[float] = None,
                 entry_gate: Optional[float] = None,
                 ramp_floor: float = 0.4,
                 ramp_hold: bool = True):
        super().__init__(loader=loader, indicator=indicator, cost=cost)
        self.ramp_score = None if ramp_score is None else float(ramp_score)
        self.full_score = None if full_score is None else float(full_score)
        self.entry_gate = None if entry_gate is None else float(entry_gate)
        self.ramp_floor = float(ramp_floor)
        self.ramp_hold = bool(ramp_hold)
        if not (0.0 < self.ramp_floor <= 1.0):
            raise ValueError(f"[satellite_v2] ramp_floor 는 (0,1] 범위여야 합니다: {self.ramp_floor}")
        if (self.ramp_score is not None and self.full_score is not None
                and self.full_score < self.ramp_score):
            raise ValueError(f"[satellite_v2] full_score({self.full_score})는 "
                             f"ramp_score({self.ramp_score}) 이상이어야 합니다.")

    # ── 슬롯 경사 ───────────────────────────────────────────────────
    def _ramp_bounds(self, entry_score: float) -> Tuple[float, float, float]:
        """(경사 하단, 만충 점수, 신규 자격 문턱)을 정한다. 미설정이면 원본 이진 게이트로 축퇴."""
        lo = self.ramp_score if self.ramp_score is not None else entry_score
        full = self.full_score if self.full_score is not None else entry_score
        gate = self.entry_gate if self.entry_gate is not None else lo
        return lo, full, gate

    def _fill(self, scores: np.ndarray, entry: float, full: float) -> np.ndarray:
        """점수 → 슬롯 충전율(0~1). entry 에서 ramp_floor, full 이상에서 1.0 인 선형 경사.

        entry 미만(= exit_score 까지 잔류 중인 보유 종목)은 하한 충전율로 둔다. `full <= entry`
        면 경사가 없다는 뜻이므로 전부 1.0 을 돌려 원본과 비트 단위로 같아진다.
        """
        if full <= entry:
            return np.ones_like(scores)
        ramp = np.clip((scores - entry) / (full - entry), 0.0, 1.0)
        return self.ramp_floor + (1.0 - self.ramp_floor) * ramp

    def _targets(self, row: np.ndarray, close_row: np.ndarray, held_now: np.ndarray,
                 top_n: int, entry_score: float, exit_score: float
                 ) -> Tuple[np.ndarray, float, np.ndarray]:
        """오늘 종가 점수로 (종목 목표비중, 현금비중, 선정 인덱스)를 만든다.

        자격 판정(히스테리시스)은 원본과 같은 꼴이다 — 미보유는 entry_gate, 보유는 exit_score.
        달라지는 건 비중뿐: 원본 `1/top_n` 고정 → `f(점수)/top_n` 경사.
        `ramp_hold=False` 면 보유 슬롯은 경사를 면제하고 가득 채운다(부분 '진입'만 실험).
        """
        lo, full, gate = self._ramp_bounds(entry_score)
        thr = np.where(held_now, exit_score, gate)
        elig = np.where(~np.isnan(row) & ~np.isnan(close_row) & (row >= thr))[0]
        ranked = elig[np.argsort(row[elig])[::-1]] if len(elig) else elig  # 점수 내림차순
        sel = ranked[:top_n]

        w_names = np.zeros(len(row))
        if not len(sel):
            return w_names, 1.0, sel                     # 자격 0개 → 전액 현금
        fills = self._fill(row[sel], lo, full)
        if not self.ramp_hold:
            fills = np.where(held_now[sel], 1.0, fills)  # 보유 슬롯은 만충 유지
        w_names[sel] = fills / top_n
        # 못 채운 슬롯 몫은 현금 대용으로. 합을 먼저 낸 뒤 나눠야 축퇴 시 원본과 오차 0.
        w_cash = 1.0 - fills.sum() / top_n
        return w_names, w_cash, sel

    # ── 시뮬레이션 ──────────────────────────────────────────────────
    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame,
                  atr: Optional[pd.DataFrame], cash_ret: pd.Series, top_n: int,
                  period: str, trailing: Optional[TrailingStop],
                  entry_score: float, exit_score: float
                  ) -> Tuple[pd.Series, List[pd.Timestamp], int, List[str],
                             List[Tuple[pd.Timestamp, List[str]]]]:
        """원본 `_simulate` 과 동일하되, 목표비중 산정(블록 d)만 경사 진입으로 바꾼다.

        원본에 훅(hook) 자리가 없어 루프를 통째로 복제했다(원본 수정 금지 원칙). 블록 (a)~(c)와
        룩어헤드·비용·고점 규약은 원본 그대로이므로, 원본이 바뀌면 이 파일도 함께 손봐야 한다.
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
        cash_val = 0.0           # 현금 대용 슬롯 가치
        peak = np.full(N, np.nan)
        holding = False
        pending: Optional[Tuple[np.ndarray, float]] = None
        eq = np.empty(T)
        rotations: List[pd.Timestamp] = []
        pick_log: List[Tuple[pd.Timestamp, List[str]]] = []
        stops = 0
        prev_set: frozenset = frozenset()

        for i in range(T):
            # (a) 당일 수익 반영(첫 투자 이후). 종목 결측 수익은 0, 현금은 단기채권 수익.
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
            # (c) 트레일링 스탑 점검(당일 종가) → 손절 슬롯을 현금으로 대피
            if holding and trailing is not None:
                held = np.where(name_val > 0.0)[0]
                for j in held:
                    if np.isnan(C[i, j]):
                        continue
                    peak[j] = C[i, j] if np.isnan(peak[j]) else max(peak[j], C[i, j])
                    atr_j = A[i, j] if A is not None else np.nan
                    if C[i, j] <= trailing.stop_level(peak[j], atr_j):
                        cash_val += name_val[j]
                        name_val[j] = 0.0
                        peak[j] = np.nan
                        stops += 1
            # (d) 오늘 종가 점수로 목표비중 산정(★ 경사 진입 — 원본과 여기만 다름)
            if check[i]:
                w_names, w_cash, sel = self._targets(
                    S[i], C[i], name_val > 0.0, top_n, entry_score, exit_score)
                pending = (w_names, w_cash)
                cur_set = frozenset(int(s) for s in sel)
                if cur_set != prev_set:                              # 보유 구성 변경 = 교체
                    rotations.append(idx[i])
                    pick_log.append((idx[i], [tickers[j] for j in sel]))
                    prev_set = cur_set

        last_pick = [tickers[j] for j in sorted(prev_set)] if prev_set else []
        return pd.Series(eq, index=idx, name="equity"), rotations, stops, last_pick, pick_log
