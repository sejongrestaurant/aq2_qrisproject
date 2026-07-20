"""롤링 수익 분포 산출물 — 보유기간별 분포 상자그림(제안서 §6 · 발표용).

**이 그림의 논지는 상자의 아래쪽 끝에 있다.** 가운데(중앙값)는 어느 상품이나 그럴듯하다.
하락 방어형의 값은 '최악의 가입 타이밍'에서 드러나므로, 보유기간이 길어질 때 **아래쪽 끝이
0 선 위로 올라오는 속도**를 서로 비교하게 그린다. 그래서 수염을 사분위가 아니라 **최저~최고
전 범위**로 두고(이상치 점을 따로 찍지 않는다), 상품의 최저값에만 숫자를 붙인다.

색·역할 규약은 다른 전시물과 공유한다(`regime_report`·`drawdown_report`): 파랑=상품,
주황=상품 벤치마크, 먹색=참고 지수.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .report_base import ReportWriter, plt
from .rolling_returns import RollingReturns

logger = logging.getLogger(__name__)

_COLORS = ["#2a78d6", "#c2681a", "#52514e"]
_C_INK = "#0b0b0b"
_C_MUTED = "#5c5b57"


class RollingReturnReport(ReportWriter):
    """롤링 수익 분포를 CSV·PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    def write_table(self, roll: RollingReturns, horizons: Sequence[int],
                    name: str = "rolling_returns") -> str:
        """대상 × 보유기간 요약표."""
        return self._write_csv(roll.table(horizons), name)

    def plot_box(self, roll: RollingReturns, horizons: Sequence[int],
                 name: str = "rolling_returns") -> str:
        """보유기간별 연율 수익률 분포(대상별 상자그림).

        Args:
            roll: 분포 계산기. `roll.curves` 의 **순서가 곧 역할**이다(상품 → 벤치 → 참고).
            horizons: 보유기간(개월) 목록.
        """
        names = list(roll.curves)
        n = len(names)
        width = 0.8 / n                       # 한 보유기간 그룹의 폭 0.8 을 대상 수로 나눈다
        fig, ax = plt.subplots(figsize=(10.5, 5.4))

        used, mins, tops = [], {}, {}
        for gi, h in enumerate(horizons):
            w = roll.windows(h)
            if w.empty:
                continue
            used.append((gi, h, len(w)))
            for si, label in enumerate(names):
                v = w[label].to_numpy(dtype=float)
                pos = gi + (si - (n - 1) / 2) * width
                tops[(pos, si)] = (float(v.min()), float(v.max()))
                bp = ax.boxplot([v], positions=[pos], widths=width * 0.82,
                                whis=(0, 100), showfliers=False, patch_artist=True,
                                medianprops=dict(color="white", lw=1.6),
                                whiskerprops=dict(color=_COLORS[si], lw=1.0),
                                capprops=dict(color=_COLORS[si], lw=1.0),
                                boxprops=dict(facecolor=_COLORS[si], edgecolor=_COLORS[si],
                                              lw=0.8, alpha=0.85))
                del bp
                if si == 0:                   # 상품의 '최악의 가입 타이밍'에만 숫자를 붙인다
                    mins[pos] = float(v.min())

        # y 범위는 **상품과 상품 벤치마크**에 맞춘다. 참고 지수(KOSPI200)의 12개월 창은
        # 2020 저점 기준 반등이 연율 +296% 까지 튀어, 그걸 다 담으면 정작 비교해야 할
        # 두 상자가 납작해진다. 잘라 내되 **잘린 최고치는 숫자로 적어** 감추지 않는다.
        main = [mx for (_, si), (_, mx) in tops.items() if si < min(2, n)]
        lows = [mn for (mn, _) in tops.values()]
        top = max(main) * 1.20
        bottom = min(lows) - (top - min(lows)) * 0.16       # 아래 '최저' 라벨 자리
        for (pos, si), (_, mx) in tops.items():
            if mx > top:
                ax.annotate(f"↑ 최고 {mx:+.0f}%", (pos, top), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=7.5, color=_COLORS[si], zorder=6)

        for pos, v in mins.items():
            # 0 선·이웃 상자와 겹칠 수 있어 흰 배경을 깐다(라벨이 이 그림의 논지다).
            ax.annotate(f"최저 {v:+.1f}%", (pos, v), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8, color=_COLORS[0], zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))

        ax.set_ylim(bottom, top)
        ax.axhline(0, color=_C_INK, lw=0.9, ls="--", alpha=0.55, zorder=1)
        ax.set_xticks([gi for gi, _, _ in used])
        ax.set_xticklabels([f"{h}개월\n(창 {k}개)" for _, h, k in used], fontsize=9)
        ax.set_xlabel("보유기간 — 모든 시작 월을 굴린 창", labelpad=8)
        ax.set_ylabel("연율 수익률(%)")
        ax.set_title("롤링 수익 분포 — 언제 가입했든, 최악의 타이밍은 얼마였나")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=c, alpha=0.85)
                   for c in _COLORS[:n]]
        # 범례는 축 밖 아래로 — 위쪽은 '축 밖 최고치' 표기가 쓰는 자리다.
        ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                  ncol=n, frameon=False, fontsize=8.5)
        ax.text(0.5, -0.36,
                "상자=사분위, 수염=최저~최고 전 범위, 흰 선=중앙값. y축은 상품·벤치 범위에 "
                "맞췄고 축 밖 최고치는 숫자로 표기. 창끼리 구간이 겹쳐 독립 표본이 아니다.",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=_C_MUTED)
        ax.grid(axis="y", color=_C_MUTED, alpha=0.18, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return self._save(fig, name)
