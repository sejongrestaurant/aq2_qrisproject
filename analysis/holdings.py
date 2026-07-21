"""월별 구성(보유 종목) 관측 — 동결 V2 가 매 체크에서 무엇을 얼마나 들었나(제안서 전시물).

**왜 이 모듈이 필요한가.** `run_exposure.py` 는 체크 시점별 *총* 노출(만충 대비 몇 %)만 낸다.
제안서·발표에서는 그 아래 한 단계, 즉 "그 노출이 어느 종목으로 어떤 비중·점수로 구성됐고
다음 달까지 각자 얼마를 벌었나"를 행 단위로 보여줄 표가 필요하다. 이 모듈이 그 표를 만든다.

**측정 방식 — 재계산하지 않고 관측한다(`ExposureProbe` 와 같은 원칙).** 점수·히스테리시스·
슬롯 경사 판정을 여기서 다시 구현하면 엔진과 미세하게 어긋난다. 이 프로브는 상위 클래스의
`_targets()` 가 **이미 정한** 선정·목표비중을 받아 적기만 하므로, 기록값은 시뮬레이션이 실제로
쓴 값과 정의상 일치한다. 구간수익은 엔진이 실제로 쓴 종가 행렬(`closes`)과 슬리브 자산곡선
(`equity`) 그 자체에서 뽑으므로 측정 창 불일치가 구조적으로 불가능하다.

**구간수익의 규약(기존 전시물과 일치).** 종목별·슬리브 구간수익은 모두 **이 체크일 → 다음
체크일** 창의 변화로 잰다 — `_build_rotations_log`(로테이션 내역)와 `contribution_analysis.py`
가 쓰는 규약 그대로다. 종목 구간수익은 그 종목 종가의 순수 가격변화(비중·현금 무관)이고,
슬리브 구간수익은 현금 대피·회전율 비용·1일 체결 지연이 모두 반영된 슬리브의 **실현** 수익이다.
따라서 종목별 수익의 비중가중 평균이 슬리브 수익과 정확히 일치하지는 않는다(서로 다른 측정치).
마지막 체크는 다음 체크가 없으므로 창을 **보유 종목이 실제로 가격을 갖는 마지막 거래일**까지
잡는다 — 캐시가 하루 더 긴 종목(금 411060) 탓에 생기는 '고아 꼬리' 날짜를 배제하기 위해서다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.schedule import period_mask
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger(__name__)

# 자격 종목이 하나도 없어 전액 현금으로 대피한 체크의 자리채움 라벨(그 체크가 표에서 사라지지 않게).
_ALL_CASH_CODE = "(현금)"
_ALL_CASH_NAME = "전액 현금 대피(자격 0)"


@dataclass(frozen=True)
class _Check:
    """체크 시점 한 번의 관측치(내부용) — 선정 인덱스·점수·슬리브내 목표비중.

    Attributes:
        sel: 선정 종목의 열 인덱스(점수 내림차순). 빈 배열이면 전액 현금.
        scores: 각 선정 종목의 체크일 TrendScore.
        name_w: 각 선정 종목의 **슬리브 내부** 목표비중(합 = 1 − 현금비중). 슬롯 만충이면 1/top_n.
        w_cash: 못 채운 슬롯 몫(현금 대용 비중).
    """
    sel: np.ndarray
    scores: np.ndarray
    name_w: np.ndarray
    w_cash: float


class HoldingsProbe(SatelliteBacktesterV2):
    """체크 시점마다 보유 구성(종목·점수·충전율·비중)을 기록하는 관측용 사테라이트 백테스터.

    엔진 로직은 일절 바꾸지 않는다 — 상위 클래스의 `_targets()` 를 그대로 호출하고 결과만
    받아 적는다. 따라서 이 프로브를 끼운 백테스트의 자산곡선은 끼우지 않은 것과 **동일**하다
    (러너가 이를 실제로 대조해 검증한다).

    Args (생성자): 상위 클래스와 동일. 동결 V2 는 `ramp_score/full_score/ramp_floor` 를 준다.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rows: List[_Check] = []
        self._checks: Optional[pd.DatetimeIndex] = None
        self._closes: Optional[pd.DataFrame] = None
        self._equity: Optional[pd.Series] = None
        self._top_n: Optional[int] = None

    # ── 관측 ────────────────────────────────────────────────────────
    def _targets(self, row: np.ndarray, close_row: np.ndarray, held_now: np.ndarray,
                 top_n: int, entry_score: float, exit_score: float
                 ) -> Tuple[np.ndarray, float, np.ndarray]:
        """상위 클래스의 선정·목표비중을 그대로 쓰고, 그 결과를 기록한다."""
        w_names, w_cash, sel = super()._targets(
            row, close_row, held_now, top_n, entry_score, exit_score)
        # sel 은 이미 점수 내림차순. 선정 종목의 점수·슬리브내 비중을 그 순서 그대로 떠 둔다.
        self._rows.append(_Check(
            sel=np.asarray(sel, dtype=int),
            scores=row[sel].copy(),
            name_w=w_names[sel].copy(),
            w_cash=float(w_cash),
        ))
        return w_names, w_cash, sel

    def _simulate(self, closes: pd.DataFrame, scores: pd.DataFrame,
                  atr: Optional[pd.DataFrame], cash_ret: pd.Series, top_n: int,
                  period: str, trailing, entry_score: float, exit_score: float):
        """상위 클래스 루프를 그대로 돌리고, 기록을 엔진의 체크일에 붙인다.

        루프를 복제하지 않는다. `_targets()` 는 체크일마다 정확히 한 번 불리므로 호출 순서가
        곧 체크일 순서다. 그 가정이 깨지면(엔진 변경 등) 조용히 어긋난 시계열을 내는 대신
        멈춘다 — 이 프로젝트에서 조용한 이탈은 이미 사고를 냈다(`ExposureProbe` 와 같은 가드).
        """
        self._rows = []
        out = super()._simulate(closes, scores, atr, cash_ret, top_n, period, trailing,
                                entry_score, exit_score)
        checks = closes.index[period_mask(closes.index, period)]
        if len(checks) != len(self._rows):
            raise RuntimeError(
                f"[holdings] 체크일 {len(checks)}개인데 관측 {len(self._rows)}개 — "
                f"엔진의 _targets() 호출 규약이 바뀐 것으로 보입니다. 측정을 신뢰할 수 없습니다.")
        self._checks = checks
        self._closes = closes
        self._equity = out[0]        # (equity, rotations, stops, last_pick, pick_log)
        self._top_n = top_n
        return out

    # ── 결과 ────────────────────────────────────────────────────────
    def holdings(self, sleeve_weight: float,
                 names: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
        """관측을 행 단위(체크일 × 보유 종목) 상세표로 낸다.

        Args:
            sleeve_weight: 포트폴리오에서 사테라이트가 갖는 비중(IRPConfig.satellite_weight, 예 0.70).
                포트폴리오 비중·위험자산 총노출을 이 비중으로 환산한다.
            names: {티커: 표시명} 맵(가독성용). 없으면 코드를 그대로 종목명으로 쓴다.
        Returns:
            컬럼: 체크일·다음체크일·코드·종목명·TrendScore·충전율%·포트폴리오비중%·
            종목구간수익%·슬리브구간수익%·위험자산총노출%. 마지막 세 컬럼(체크 단위 합계)은
            같은 체크의 모든 종목 행에 동일 값으로 반복된다(피벗·집계 편의). 자격 0 인 체크는
            '(현금)' 자리채움 행 하나로 남겨 체크 자체가 표에서 누락되지 않게 한다.
        Raises:
            RuntimeError: 백테스트를 돌리기 전에 부른 경우.
        """
        if self._checks is None or self._closes is None or self._equity is None:
            raise RuntimeError("[holdings] run() 을 먼저 호출해야 합니다.")
        names = names or {}
        idx = self._equity.index                       # 슬리브·종가와 공통 거래일 축
        C = self._closes.to_numpy(dtype=float)
        eq = self._equity.to_numpy(dtype=float)
        tickers = list(self._closes.columns)
        checks = self._checks
        top_n = int(self._top_n)

        # 각 체크의 다음 체크일(마지막 체크는 마지막 거래일)까지가 구간 창.
        pos = [idx.get_loc(d) for d in checks]
        logger.info(f"보유 구성 관측 · 체크 {len(checks)}회 · "
                    f"{checks[0]:%Y-%m-%d}~{checks[-1]:%Y-%m-%d}")

        rows: List[dict] = []
        for k, rec in enumerate(self._rows):
            pa = pos[k]
            pb = pos[k + 1] if k + 1 < len(pos) else len(idx) - 1
            date = checks[k]
            # 보유 종목별 창 [pa, pb] 내 마지막 유효 종가 인덱스. 선정 종목은 체크일(pa) 종가가
            # 유효함이 보장되므로(엔진 elig 조건) 최소 pa 는 잡힌다.
            last_fin = {int(j): pa + int(np.where(np.isfinite(C[pa:pb + 1, j]))[0][-1])
                        for j in rec.sel}
            # 유효 구간 종료 인덱스: 보유 종목이 실제로 가격을 갖는 마지막 날. 마지막 체크의 창이
            # 캐시가 하루 더 긴 종목(예 금 411060) 때문에 나머지 보유 종목은 값이 없는 '고아 꼬리'
            # 날짜로 끝나는 것을 막는다(contribution_analysis 와 같은 방어). 슬리브·종목 구간수익을
            # 같은 창에서 재도록 슬리브 수익도 이 인덱스까지만 본다.
            pb_eff = max(last_fin.values()) if last_fin else pb
            nxt = idx[pb_eff]
            # 슬리브 구간수익(실현): 슬리브 자산곡선의 이 체크 → 유효 구간 종료일 변화.
            sleeve_ret = (eq[pb_eff] / eq[pa] - 1.0) if pb_eff > pa else 0.0
            fill = float(rec.name_w.sum())             # 슬리브 충전율(= 1 − 현금비중)
            total_exposure = fill * sleeve_weight       # 위험자산 총노출(포트폴리오 기준)

            common = {
                "다음체크일": nxt.date(),
                "슬리브구간수익%": round(sleeve_ret * 100, 2),
                "위험자산총노출%": round(total_exposure * 100, 2),
            }
            if len(rec.sel) == 0:
                # 자격 0 → 전액 현금. 체크가 표에서 사라지지 않게 자리채움 한 행을 남긴다.
                rows.append({"체크일": date.date(), "코드": _ALL_CASH_CODE,
                             "종목명": _ALL_CASH_NAME, "TrendScore": np.nan,
                             "충전율%": 0.0, "포트폴리오비중%": 0.0,
                             "종목구간수익%": np.nan, **common})
                continue
            for j, score, w in zip(rec.sel, rec.scores, rec.name_w):
                code = tickers[j]
                # 종목 구간수익: 그 종목 종가의 순수 가격변화(비중·현금 무관). 창은 슬리브와 동일하되
                # 종목 자신의 마지막 유효 종가까지 본다. 체크일 뒤 값이 하나도 없으면(lj==pa) 미상(NaN).
                lj = last_fin[int(j)]
                seg = (C[lj, j] / C[pa, j] - 1.0) if lj > pa else np.nan
                rows.append({
                    "체크일": date.date(),
                    "코드": code,
                    "종목명": names.get(code, code),
                    "TrendScore": round(float(score), 2),
                    # 충전율: 슬롯 만충(1/top_n) 대비 실제 채운 비율(30~100). 경사 진입의 정체성.
                    "충전율%": round(float(w) * top_n * 100, 2),
                    "포트폴리오비중%": round(float(w) * sleeve_weight * 100, 2),
                    "종목구간수익%": round(seg * 100, 2) if np.isfinite(seg) else np.nan,
                    **common,
                })
        return pd.DataFrame(rows, columns=[
            "체크일", "다음체크일", "코드", "종목명", "TrendScore", "충전율%",
            "포트폴리오비중%", "종목구간수익%", "슬리브구간수익%", "위험자산총노출%"])
