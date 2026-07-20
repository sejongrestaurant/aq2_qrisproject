"""벤치마크 병기 비교 — 절대지표 + 상대지표(추적오차·정보비율).

**왜 KOSPI200 을 함께 놓나.** 상품 벤치마크는 KODEX TRF7030(선진국주식70/국내채권30)이다.
자산배분이 같은 급이라 비교가 공정하기 때문이다. 그러나 IRP 가입자가 실제로 머릿속에 갖고
있는 잣대는 '국내 주식시장'(KOSPI200)이다. 두 잣대를 나란히 놓아야 "무엇 대비 방어인가"가
분명해진다 — TRF7030 대비로는 같은 위험군 안에서의 우열이고, KOSPI200 대비로는 주식
전량 보유 대비 얼마나 덜 잃었나다.

**상대지표는 무엇을 말하나.**
  · 추적오차(TE) = 일간 초과수익의 표준편차 연율화. 벤치마크에서 얼마나 멀리 떨어져
    움직이나 = **액티브의 크기**. 액티브 ETF 는 이 값이 커야 정상이고, 작으면 인덱스에
    수수료만 얹은 상품이라는 뜻이다.
  · 정보비율(IR) = 연율화 초과수익 / TE. 그 이탈 한 단위로 초과수익을 얼마나 벌었나 =
    **액티브의 효율**. TE 가 큰 것 자체는 자랑이 아니고, IR 이 그 크기를 정당화한다.
  · 베타·상관은 이탈의 성격을 말해 준다(방향은 같은데 진폭이 작은가, 아예 다른 궤적인가).

**지표 정의는 재구현하지 않는다.** CAGR·Sharpe·MDD·Calmar 는 `BacktestResult._curve_metrics`
를 그대로 불러 쓴다. 여기서 따로 구현하면 리포트의 수치와 미세하게 어긋난 표가 제안서에
실릴 수 있다(같은 이유로 `analysis/exposure.py` 도 엔진 판정을 재구현하지 않는다).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from backtest import BacktestResult

logger = logging.getLogger(__name__)

_ANN = 252  # 연율화 기준 거래일 수(BacktestResult 와 동일 규약)


def align_curve(curve: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """외부 곡선(벤치마크 종가 등)을 전략 날짜축에 맞춰 시작 1.0 으로 정규화한다.

    전략 거래일에 맞춰 ffill 한 뒤 첫날로 나눈다. 창을 전략 축에 강제로 맞추는 이유는
    구 `universe_availability.py` 사고와 같다 — 창이 하루라도 어긋나면 수익률이 조용히
    달라진 표가 나온다.

    Raises:
        ValueError: 구간 안에 값이 하나도 없거나 첫 값이 결측인 경우.
    """
    aligned = curve.reindex(index).ffill()
    if aligned.isna().all() or pd.isna(aligned.iloc[0]) or float(aligned.iloc[0]) == 0.0:
        raise ValueError("[benchmark] 전략 구간에 벤치마크 값이 없습니다(창 불일치·상장 이후 확인).")
    return (aligned / float(aligned.iloc[0])).rename(curve.name)


@dataclass(frozen=True)
class RelativeStats:
    """전략 − 벤치마크 상대지표 한 쌍.

    Attributes:
        te_pct: 추적오차(연율화 %). 일간 초과수익 표준편차 × √252.
        ir: 정보비율 = 연율화 초과수익 / TE. TE=0 이면 0.
        excess_cagr_pct: CAGR 차이(전략 − 벤치, %p). IR 분자와 달리 기하 기준이라
            산술 초과수익과 미세하게 다르다 — 둘 다 내되 해석은 이 값으로 한다.
        corr: 일간수익 상관계수.
        beta: 벤치마크 대비 베타(공분산 / 벤치 분산).
    """
    te_pct: float
    ir: float
    excess_cagr_pct: float
    corr: float
    beta: float


class BenchmarkComparison:
    """전략 + 벤치마크 여러 개를 한 표로 놓는 비교기.

    Args (생성자):
        strategy: 전략 자산곡선(시작 1.0). 상대지표의 기준이 되는 곡선.
        benchmarks: {표시명: 자산곡선}. 전략과 **같은 날짜축**이어야 한다
            (`align_curve()` 로 맞춰 넣을 것).
        strategy_label: 표에 쓸 전략 표시명.

    Raises:
        ValueError: 벤치마크 날짜축이 전략과 다른 경우(창 불일치는 조용히 틀린 표를 만든다).
    """

    def __init__(self, strategy: pd.Series, benchmarks: Mapping[str, pd.Series],
                 strategy_label: str = "HELM(동결 V2)"):
        self.strategy = strategy
        self.benchmarks = dict(benchmarks)
        self.strategy_label = strategy_label
        for name, curve in self.benchmarks.items():
            if not curve.index.equals(strategy.index):
                raise ValueError(f"[benchmark] '{name}' 날짜축이 전략과 다릅니다 "
                                 f"({len(curve)} vs {len(strategy)}) — align_curve() 로 맞추세요.")

    # ── 절대지표 ────────────────────────────────────────────────────
    def absolute(self) -> pd.DataFrame:
        """전략·벤치마크 각각의 표준 성과지표표(CAGR·Sharpe·MDD·Calmar).

        제안서 §8 표의 '벤치 컷 Calmar' 빈칸을 채우는 것이 이 표다 — 전체 구간뿐 아니라
        2025년 말 컷에서도 같은 잣대로 벤치마크를 재야 우열 주장이 두 구간 모두에서 성립한다.
        """
        rows = []
        for label, curve in [(self.strategy_label, self.strategy), *self.benchmarks.items()]:
            m = BacktestResult._curve_metrics(curve, None, None)  # 정의 일치 위해 엔진 것 재사용
            rows.append({
                "대상": label,
                "CAGR%": round(m["cagr_pct"], 2),
                "Sharpe": round(m["sharpe"], 2),
                "MDD%": round(m["mdd_pct"], 2),
                "Calmar": round(m["calmar"], 2),
                "Sortino": round(m["sortino"], 2),
                "총수익%": round(m["total_return_pct"], 1),
            })
        return pd.DataFrame(rows)

    # ── 상대지표 ────────────────────────────────────────────────────
    def relative(self) -> pd.DataFrame:
        """벤치마크별 추적오차·정보비율·초과 CAGR·상관·베타 표."""
        rows = []
        for name, curve in self.benchmarks.items():
            r = self.relative_stats(curve)
            rows.append({
                "벤치마크": name,
                "초과CAGR%p": round(r.excess_cagr_pct, 2),
                "추적오차%": round(r.te_pct, 2),
                "정보비율": round(r.ir, 2),
                "상관": round(r.corr, 2),
                "베타": round(r.beta, 2),
            })
        return pd.DataFrame(rows)

    def relative_stats(self, benchmark: pd.Series) -> RelativeStats:
        """한 벤치마크에 대한 상대지표를 계산한다.

        초과수익은 **일간 산술차**(전략수익 − 벤치수익)로 잡는다. 로그수익 차·기하 차 등
        변형이 있으나 업계 관행(GIPS 계열 보고)이 산술차이고, 분모(TE)와 분자를 같은
        정의로 맞춰야 IR 이 해석 가능하다.
        """
        rs = self.strategy.pct_change().fillna(0.0)
        rb = benchmark.pct_change().fillna(0.0)
        diff = rs - rb

        te = float(diff.std() * np.sqrt(_ANN))
        ir = float(diff.mean() * _ANN / te) if te > 0 else 0.0
        var_b = float(rb.var())
        beta = float(rs.cov(rb) / var_b) if var_b > 0 else 0.0

        ms = BacktestResult._curve_metrics(self.strategy, None, None)
        mb = BacktestResult._curve_metrics(benchmark, None, None)
        return RelativeStats(
            te_pct=te * 100,
            ir=ir,
            excess_cagr_pct=ms["cagr_pct"] - mb["cagr_pct"],
            corr=float(rs.corr(rb)),
            beta=beta,
        )

    # ── 로그 ────────────────────────────────────────────────────────
    def summary_lines(self) -> list[str]:
        """콘솔용 사람 읽는 요약(폭 지정 정렬)."""
        lines = [f"{'대상':<18}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'Calmar':>8}"]
        for _, r in self.absolute().iterrows():
            lines.append(f"{r['대상']:<18}{r['CAGR%']:>8.1f}{r['Sharpe']:>8.2f}"
                         f"{r['MDD%']:>8.1f}{r['Calmar']:>8.2f}")
        lines.append("")
        lines.append(f"{'벤치마크':<18}{'초과CAGR%p':>11}{'TE%':>8}{'IR':>7}{'상관':>7}{'베타':>7}")
        for _, r in self.relative().iterrows():
            lines.append(f"{r['벤치마크']:<18}{r['초과CAGR%p']:>11.2f}{r['추적오차%']:>8.2f}"
                         f"{r['정보비율']:>7.2f}{r['상관']:>7.2f}{r['베타']:>7.2f}")
        return lines
