"""롤링 수익 분포(거치식) — "언제 가입했든" 얼마를 벌었나.

**왜 필요한가.** 단일 구간 CAGR 13.4% 는 2020-01-02 라는 **시작일 하나에 걸린 우연**이다.
IRP 가입자는 자기가 가입한 달에 시작할 뿐 시작일을 고를 수 없다. 그래서 필요한 건 한 점의
수익률이 아니라 모든 시작 월의 **분포**이고, 특히 분포의 **아래쪽 끝**(최악의 가입 타이밍)이다.
하락 방어형 상품의 값은 평균이 아니라 최악에서 드러난다.

`analysis/rolling.py` 와 무엇이 다른가: 그쪽은 **적립식**(월 납입 현금흐름 대비 손익률),
이쪽은 **거치식**(한 번에 넣고 N개월 보유했을 때의 연율 수익률)이다. 제안서에서 전자는
§8 적립식 시나리오, 후자는 §6 백테스트 성과에 들어간다. 둘을 한 그림에 섞으면 분모가
다른 수치가 같은 축에 놓인다.

**한계(반드시 병기).** 구간이 78개월뿐이라 보유기간이 길수록 창이 급감하고(60개월 → 19창)
창끼리 구간이 겹쳐 **독립 표본이 아니다**. 여기 나오는 '손실 확률'은 이 6.5년 안에서 시작
시점을 굴렸을 때의 빈도이지 미래 확률의 추정치가 아니다.
"""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RollingReturns:
    """자산곡선 여러 개를 같은 롤링 창으로 재는 분포 계산기.

    Args (생성자):
        curves: {표시명: 자산곡선}. 모두 같은 날짜축이어야 한다(창이 어긋나면 비교가 깨진다).

    Raises:
        ValueError: 곡선들의 날짜축이 서로 다른 경우.
    """

    def __init__(self, curves: Mapping[str, pd.Series]):
        self.curves = dict(curves)
        idx = next(iter(self.curves.values())).index
        for name, c in self.curves.items():
            if not c.index.equals(idx):
                raise ValueError(f"[rolling_returns] '{name}' 날짜축이 다릅니다 — 같은 축으로 맞추세요.")
        self.index = idx
        # 각 달의 첫 거래일 = 가능한 가입 시점(투자자는 월 단위로 가입한다).
        self.month_starts = (pd.Series(idx, index=idx)
                             .groupby([idx.year, idx.month]).min().to_numpy())

    # ── 분포 ────────────────────────────────────────────────────────
    def windows(self, horizon_months: int) -> pd.DataFrame:
        """보유기간별 창 수익률(연율 %) — 열=대상, 행=시작 월.

        창의 끝이 데이터 밖이면 그 시작점은 버린다(미완성 창을 섞으면 보유기간이 짧은
        창이 몰래 끼어든다).
        """
        rows, starts = [], []
        for s in self.month_starts:
            s = pd.Timestamp(s)
            e = s + pd.DateOffset(months=horizon_months)
            if e > self.index[-1]:
                continue
            seg_end = self.index[self.index <= e][-1]
            rows.append({name: self._annualized(c.loc[s], c.loc[seg_end], horizon_months)
                         for name, c in self.curves.items()})
            starts.append(s)
        return pd.DataFrame(rows, index=pd.DatetimeIndex(starts))

    def table(self, horizons: Sequence[int]) -> pd.DataFrame:
        """대상 × 보유기간 요약표(창 수·최저·사분위·중앙값·최고·손실 창 비율)."""
        out = []
        for h in horizons:
            w = self.windows(h)
            if w.empty:
                logger.warning(f"보유 {h}개월: 표본 창이 없어 건너뜁니다(구간 부족).")
                continue
            for name in self.curves:
                v = w[name].to_numpy(dtype=float)
                out.append({
                    "보유개월": h,
                    "대상": name,
                    "창수": len(v),
                    "최저%": round(float(v.min()), 2),
                    "25%": round(float(np.percentile(v, 25)), 2),
                    "중앙값%": round(float(np.median(v)), 2),
                    "75%": round(float(np.percentile(v, 75)), 2),
                    "최고%": round(float(v.max()), 2),
                    "손실창비율%": round(float((v < 0).mean() * 100), 1),
                })
        return pd.DataFrame(out)

    @staticmethod
    def _annualized(v0: float, v1: float, horizon_months: int) -> float:
        """구간 수익을 연율(%)로 환산. 12개월 미만도 같은 규약으로 연율화한다."""
        total = float(v1) / float(v0)
        return (total ** (12.0 / horizon_months) - 1.0) * 100.0

    # ── 로그 ────────────────────────────────────────────────────────
    def summary_lines(self, horizons: Sequence[int]) -> list:
        """콘솔용 요약(보유기간 블록별, 폭 지정 정렬)."""
        tbl = self.table(horizons)
        lines = []
        for h in horizons:
            blk = tbl[tbl["보유개월"] == h]
            if blk.empty:
                continue
            lines.append(f"[보유 {h}개월 · 창 {int(blk.iloc[0]['창수'])}개]")
            lines.append(f"  {'대상':<18}{'최저%':>8}{'중앙값%':>9}{'최고%':>8}{'손실창%':>9}")
            for _, r in blk.iterrows():
                lines.append(f"  {r['대상']:<18}{r['최저%']:>8.1f}{r['중앙값%']:>9.1f}"
                             f"{r['최고%']:>8.1f}{r['손실창비율%']:>9.1f}")
            lines.append("")
        return lines
