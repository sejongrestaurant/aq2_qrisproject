"""A-2 채권 슬리브 월별 배분 관측 — 채권 스위칭이 매 체크에서 듀레이션을 어떻게 나눴나(전시물).

`analysis.holdings.HoldingsProbe` 를 **그대로 상속**해 채권 슬리브(153130/114260/439870, top_n=3)에
끼운다. 엔진 로직(선정·경사·히스테리시스)은 일절 재구현하지 않는다 — 상위 프로브가 이미 관측한
`_Check`(선정·슬리브내 비중·현금몫)를 받아 적기만 한다. 딱 하나 더 관측하는 것: **선정되지 않은
채권의 점수**다(3종 전체의 TrendScore 를 매 체크 보여주려면 랭킹에서 빠진 채권 점수도 필요하다).
그 값도 엔진이 이미 계산해 넘긴 점수행(`row`)에서 떠 오므로 재계산이 아니다.

비중 규약(합 100%): 채권 슬리브의 현금 대용이 곧 **단기채 153130** 이므로, 못 채운 슬롯 몫
(`w_cash`)은 153130 의 비중에 접어 넣는다. 그러면 세 채권 비중의 합이 정확히 100% 가 되어 '슬리브
30% 안에서 듀레이션을 어떻게 나눴나'를 그대로 읽을 수 있다.

창 절단 방어: 439870(국고30)은 2022-08 상장이라 그 이전 체크에선 점수 NaN·비중 0 으로 남는다
(엔진이 상장 전 후보에서 자동 제외 — 슬리브가 흡수). 상위 HoldingsProbe 의 고아꼬리(411060) 방어도
그대로 상속되나, 이 표는 순수 비중·점수만 내보내고 전방 구간수익을 계산하지 않으므로 트리거되지
않는다(채권 슬리브에 411060 자체가 없다).
"""
from __future__ import annotations

import logging
from typing import List, Mapping

import numpy as np
import pandas as pd

from analysis.holdings import HoldingsProbe

logger = logging.getLogger(__name__)


class BondSleeveProbe(HoldingsProbe):
    """채권 슬리브 관측 프로브 — 상위 관측 + '전체 채권 점수행'만 추가로 떠 둔다.

    Args (생성자): 상위와 동일(동결 경사 인자 `ramp_score/full_score/ramp_floor/ramp_hold`).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._score_rows: List[np.ndarray] = []  # 체크별 전체 점수행(선정 밖 채권 포함)

    def _targets(self, row, close_row, held_now, top_n, entry_score, exit_score):
        # 엔진이 넘긴 점수행을 그대로 떠 둔 뒤 상위 관측(_Check 기록)에 위임 — 순서=체크순.
        self._score_rows.append(np.asarray(row, dtype=float).copy())
        return super()._targets(row, close_row, held_now, top_n, entry_score, exit_score)

    def _simulate(self, closes, scores, atr, cash_ret, top_n, period, trailing,
                  entry_score, exit_score):
        self._score_rows = []  # 상위가 self._rows 를 리셋하듯 여기서 점수행도 리셋
        return super()._simulate(closes, scores, atr, cash_ret, top_n, period, trailing,
                                 entry_score, exit_score)

    # ── 결과: 월별 채권 배분(wide) ───────────────────────────────────
    def bond_weights(self, bond_universe: List[str], cash_ticker: str,
                     short_names: Mapping[str, str]) -> pd.DataFrame:
        """체크일 × [각 채권 슬리브내 비중%(합100) · 각 채권 TrendScore] wide 표.

        Args:
            bond_universe: 채권 후보 티커(표 컬럼 순서). 예 [153130,114260,439870].
            cash_ticker: 못 채운 슬롯 몫을 접어 넣을 현금 대용 티커(= 153130).
            short_names: {티커: 짧은 표시명}(컬럼 라벨용, 예 '단기채').
        Returns:
            컬럼: 체크일 · <명(코드)>_비중% ×N · <명(코드)>_점수 ×N. 비중 합 = 100%.
        Raises:
            RuntimeError: run() 전에 부른 경우, 또는 관측 정합성이 깨진 경우.
        """
        if self._checks is None or self._closes is None:
            raise RuntimeError("[bond_holdings] run() 을 먼저 호출해야 합니다.")
        if not (len(self._checks) == len(self._rows) == len(self._score_rows)):
            raise RuntimeError("[bond_holdings] 체크·관측·점수행 길이 불일치 — 측정 신뢰 불가.")

        tickers = list(self._closes.columns)
        col = {t: tickers.index(t) for t in bond_universe}

        out: List[dict] = []
        for date, rec, srow in zip(self._checks, self._rows, self._score_rows):
            # 선정 채권의 슬리브내 비중을 티커로 매핑 후, 현금몫을 현금 티커(153130)에 접어 넣는다.
            w = {t: 0.0 for t in bond_universe}
            for j, wn in zip(rec.sel, rec.name_w):
                w[tickers[int(j)]] = float(wn)
            w[cash_ticker] += float(rec.w_cash)
            if abs(sum(w.values()) - 1.0) > 1e-9:   # 합 100% 정합 가드
                raise RuntimeError(f"[bond_holdings] {date.date()} 비중 합 {sum(w.values()):.6f} ≠ 1.")

            r = {"체크일": date.date()}
            for t in bond_universe:
                r[f"{short_names.get(t, t)}({t})_비중%"] = round(w[t] * 100, 2)
            for t in bond_universe:
                s = srow[col[t]]
                r[f"{short_names.get(t, t)}({t})_점수"] = round(float(s), 2) if np.isfinite(s) else np.nan
            out.append(r)
        return pd.DataFrame(out)
