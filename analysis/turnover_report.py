"""회전율 산출물 — 연도별 회전율·비용 드래그 차트(제안서 §6.4 · 발표용).

**이 그림이 답하는 질문은 하나다: "비용은 반영했나?"** 그래서 회전율(막대)과 그 비용(막대 위
숫자)을 같은 자리에 붙여 둔다. 회전율만 크게 그리면 "많이 돌린다"는 인상만 남고, 정작 그게
수익률 몇 %p 인지가 빠진다 — 이 상품에서는 그 답(연 0.34%p)이 오히려 방어 논리다.

**한 축만 쓴다.** 비용 드래그는 회전율 × 거래비용률의 **단위 환산**일 뿐 다른 측정량이 아니므로,
두 번째 y축을 세우지 않고 막대 위 텍스트로 붙인다(축이 둘이면 독자가 두 개의 다른 양을
비교한다고 오해한다).

**두 계층을 쌓되 흰 테두리로 가른다.** 슬리브 로테이션(월간)과 상위 리밸런싱(분기)은 기준
금액이 달라 원래는 더할 수 없지만, 포트폴리오 기준으로 환산해 두면 합계가 곧 '포트폴리오가
한 해 동안 갈아엎은 양'이 된다. 상위 리밸런싱은 슬리브의 1/70 수준이라 눈에 거의 안 보이는데,
**그 작음 자체가 정보**다(비용의 거의 전부가 월간 로테이션에서 나온다).
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import pandas as pd

from .report_base import ReportWriter, plt
from .turnover import TurnoverStats

logger = logging.getLogger(__name__)

# 색 — 파랑=주된 계층(슬리브 로테이션), 주황=부차 계층(상위 리밸런싱).
# 두 색은 팔레트 검사 6항목을 통과한다(CVD ΔE 26.3). regime_report 와 같은 슬롯.
_C_MAIN = "#2a78d6"
_C_SUB = "#c2681a"
_C_INK = "#0b0b0b"
_C_MUTED = "#5c5b57"


class TurnoverReport(ReportWriter):
    """회전율 결과를 PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    def plot_by_year(self, stats: Sequence[TurnoverStats], cost: float,
                     name: str = "turnover_by_year") -> str:
        """연도별 포트폴리오 회전율(계층 누적 막대) + 막대 위 비용 드래그.

        Args:
            stats: 계층별 집계. 첫 항목이 주 계층(파랑)으로 그려진다.
            cost: 왕복 거래비용 비율(막대 위 드래그 환산과 각주에 쓴다).
        """
        cols = {s.label: s.by_year()["회전율%"] for s in stats}
        df = pd.DataFrame(cols).fillna(0.0).sort_index()
        years = [int(y) for y in df.index]

        fig, ax = plt.subplots(figsize=(10, 5.4))
        colors = [_C_MAIN, _C_SUB]
        bottom = pd.Series(0.0, index=df.index)
        for i, label in enumerate(df.columns):
            # 흰 테두리 = 세그먼트 사이 간격. 색만으로 붙어 보이지 않게 한다.
            ax.bar(years, df[label], bottom=bottom, width=0.62,
                   color=colors[i % len(colors)], edgecolor="white", lw=1.4,
                   label=label, zorder=3)
            bottom = bottom + df[label]

        total = df.sum(axis=1)
        for x, v in zip(years, total):
            ax.annotate(f"{v:,.0f}%\n{v * cost:.2f}%p", (x, v), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=_C_INK, linespacing=1.35, zorder=4)

        # 연평균 기준선 — **양쪽 끝의 부분 연도를 뺀** 온전한 해만으로 낸다. 측정 구간이
        # 2월에 시작하고 6월에 끝나므로 첫·마지막 해를 함께 넣으면 평균이 아래로 눌린다.
        first_partial, last_partial = self._partial_ends(stats)
        full = total.iloc[(1 if first_partial else 0):(-1 if last_partial else None)]
        avg = float(full.mean())
        ax.axhline(avg, color=_C_MUTED, lw=0.9, ls=":", zorder=2)
        # 라벨은 **가장 낮은 막대 위**에 얹는다 — 그 자리가 기준선과 막대 사이가 가장 넓어
        # 어느 해가 최저인지 바뀌어도 겹치지 않는다(고정 좌표는 데이터가 바뀌면 부딪힌다).
        x_free = years[int(total.to_numpy().argmin())]
        # 그래도 이웃 막대에 걸릴 수 있으니 흰 배경을 깔아 가독성을 보장한다.
        ax.annotate(f"온전한 해({len(full)}개) 평균 {avg:,.0f}% · 비용 {avg * cost:.2f}%p",
                    (x_free, avg), xytext=(0, 5), textcoords="offset points",
                    fontsize=8.5, color=_C_MUTED, va="bottom", ha="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78))

        labels = [str(y) for y in years]
        if first_partial:
            labels[0] = f"{years[0]}\n(2월~)"
        if last_partial:
            labels[-1] = f"{years[-1]}\n(상반기)"     # 부분 연도를 온전한 해와 나란히 읽지 않게
        ax.set_xticks(years)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, float(total.max()) * 1.22)
        ax.set_ylabel("연간 회전율(단방향, 포트폴리오 기준 %)")
        ax.set_title(f"연도별 회전율과 비용 드래그 — 왕복 거래비용 {cost * 100:.2f}% 가정")
        # 범례는 축 아래로 — 막대가 대체로 높아 축 안에 두면 값 라벨과 부딪힌다.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
                  fontsize=8.5, title="막대 위 = 회전율 · 비용 드래그", title_fontsize=8.5)
        ax.text(0.5, -0.30,
                "회전율은 단방향(100% = 포트폴리오를 한 번 갈아엎음). "
                "비용 드래그는 이미 백테스트 수익률에 반영된 값이다(추가 차감 아님).",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=_C_MUTED)
        ax.grid(axis="y", color=_C_MUTED, alpha=0.18, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        return self._save(fig, name)

    # ── 내부 ────────────────────────────────────────────────────────
    @staticmethod
    def _partial_ends(stats: Sequence[TurnoverStats]) -> tuple:
        """(첫 해가 부분 연도인가, 마지막 해가 부분 연도인가).

        측정 구간이 1월에 시작하지 않거나 12월에 끝나지 않으면 그 해는 온전하지 않다.
        평균에 섞으면 회전율이 실제보다 낮아 보인다(백테스트 시작은 2020-02, 끝은 2026-06).
        """
        idx: List[pd.Timestamp] = [t for s in stats if len(s.turnover)
                                   for t in (s.turnover.index[0], s.turnover.index[-1])]
        if not idx:
            return False, False
        return min(idx).month > 1, max(idx).month < 12
