"""언더워터(낙폭) 곡선 — 낙폭의 **깊이와 지속**을 함께 보이는 전시물.

**왜 MDD 한 숫자로는 부족한가.** MDD 는 '가장 깊었던 한 점'이라 얼마나 오래 물려 있었는지를
말해 주지 않는다. 적립식 가입자가 실제로 견디는 것은 깊이보다 **지속**이다 — −20% 를 한 달
겪는 것과 −12% 로 2년을 물려 있는 것은 전혀 다른 경험이다. 언더워터 곡선은 그 두 축을
한 장에 담는다: 아래로 얼마나 파였는가(깊이) × 0 선에 언제 돌아왔는가(지속).

**그려지는 값.** 각 곡선의 `자산 / 그때까지의 최고점 − 1`(≤0). 0 은 신고가 갱신 중이라는 뜻이다.
상품(HELM)만 면으로 채우고 벤치마크는 선으로 둔다 — 셋 다 채우면 겹쳐서 아무것도 안 읽힌다.

**색 규약은 다른 전시물과 공유한다**(`regime_report`·`exposure_report`): 파랑=상품,
주황=상품 벤치마크(TRF7030), 먹색 점선=참고 지수(KOSPI200). 앞의 두 색은 팔레트 검사
6항목을 통과한 조합이고, 먹색은 '동급 시리즈가 아닌 참고선'이라는 신호로 채도를 뺀 것이다.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import pandas as pd

from backtest import BacktestResult

from .report_base import ReportWriter, plt

logger = logging.getLogger(__name__)

_C_MAIN = "#2a78d6"
_C_BENCH = "#c2681a"
_C_REF = "#52514e"
_C_MUTED = "#5c5b57"


def underwater(equity: pd.Series) -> pd.Series:
    """언더워터 곡선(%) — 그때까지의 최고점 대비 낙폭(≤0)."""
    return (equity / equity.cummax() - 1.0) * 100.0


def longest_underwater(equity: pd.Series) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """가장 오래 물려 있던 구간 (직전 고점일, 회복일)을 찾는다.

    아직 회복하지 못한 채 구간이 끝나면 그 구간은 **완결되지 않았으므로 제외**한다
    (`BacktestResult._recovery_stats` 의 규약과 같다 — 미완결 구간을 섞으면 '최장'이
    측정 종료일에 의존하는 값이 된다).
    """
    under = equity < equity.cummax()
    best: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    peak = equity.index[0]
    in_dd = False
    for date, is_under in under.items():
        if is_under:
            in_dd = True
            continue
        if in_dd:                                   # 고점 회복 → 구간 완결
            if best is None or (date - peak) > (best[1] - best[0]):
                best = (peak, date)
            in_dd = False
        peak = date
    return best


class DrawdownReport(ReportWriter):
    """언더워터 곡선을 PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    def plot_underwater(self, curves: Mapping[str, pd.Series],
                        name: str = "underwater") -> str:
        """언더워터 곡선 — 첫 곡선이 상품(면), 나머지는 벤치마크(선).

        Args:
            curves: {표시명: 자산곡선}. **순서가 곧 역할**이다(상품 → 벤치 → 참고 지수).
        """
        names = list(curves)
        styles = [(_C_MAIN, "-", 2.0), (_C_BENCH, "-", 1.5), (_C_REF, "--", 1.3)]

        fig, ax = plt.subplots(figsize=(11, 5.0))
        depth_min = 0.0
        for i, label in enumerate(names):
            color, ls, lw = styles[i % len(styles)]
            uw = underwater(curves[label])
            m = BacktestResult._curve_metrics(curves[label], None, None)  # 지표 정의 일치
            rec = f" · 최장 {m['max_recovery_days']:,.0f}일" if m["max_recovery_days"] else ""
            if i == 0:
                ax.fill_between(uw.index, uw.to_numpy(), 0.0, color=color, alpha=0.16, lw=0,
                                zorder=2)
            ax.plot(uw.index, uw.to_numpy(), color=color, ls=ls, lw=lw, zorder=3 + (i == 0),
                    label=f"{label} · MDD {m['mdd_pct']:.1f}%{rec}")
            # 최저점 표식 — '가장 깊었던 한 점'이 어디였는지 눈으로 짚어 준다.
            ax.plot([uw.idxmin()], [uw.min()], marker="o", ms=5, color=color, zorder=6)
            depth_min = min(depth_min, float(uw.min()))

        # 상품의 최장 언더워터 구간 — 깊이 말고 '지속' 축을 눈에 보이게 한다.
        span = longest_underwater(curves[names[0]])
        if span is not None:
            t0, t1 = span
            ax.axvspan(t0, t1, color=_C_MAIN, alpha=0.07, lw=0, zorder=1)
            # 곡선 위에 얹히므로 흰 배경으로 가독성을 확보한다(음영만으로는 글자가 묻힌다).
            ax.annotate(f"{names[0]} 최장 언더워터 {(t1 - t0).days:,}일 "
                        f"({t0:%Y-%m} → {t1:%Y-%m})",
                        (t0 + (t1 - t0) / 2, depth_min * 0.045), ha="center", va="top",
                        fontsize=8.5, color=_C_MUTED, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82))

        ax.axhline(0, color=_C_MUTED, lw=0.8, alpha=0.6, zorder=2)
        ax.set_ylim(depth_min * 1.16, 0)          # 0 을 위쪽 끝에 고정(아래로 파이는 그림)
        ax.margins(x=0.01)
        ax.set_ylabel("고점 대비 낙폭(%)")
        ax.set_title("언더워터 곡선 — 얼마나 깊이, 얼마나 오래 물려 있었나")
        ax.legend(loc="lower left", frameon=False, fontsize=8.5)
        ax.text(0.5, -0.15, "0 = 신고가 갱신 중. 점 = 각 곡선의 최대낙폭 지점. "
                            "'최장'은 고점 회복까지 걸린 최대 달력일수(미회복 구간 제외).",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=_C_MUTED)
        ax.grid(axis="y", color=_C_MUTED, alpha=0.18, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return self._save(fig, name)
