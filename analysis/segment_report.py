"""구간 손익 산출물 — 시기별 구간수익 × 슬롯 수 차트(제안서 §8 · 발표용).

**한 장에 두 질문을 담는다.**
  (위) *언제* 벌고 잃었나 — 교체 구간마다 그 구간의 수익을 막대로. 막대 색이 그때 **슬롯을
       몇 개 채웠는지**(=국면 대응의 강도)라서, 위기에 색이 옅어지며 막대가 짧아지는 모습이
       "노출을 줄여서 덜 잃었다"는 서사와 그림에서 겹친다. 두 번째 y축을 세우지 않고 색으로
       슬롯 수를 얹는 이유가 이것이다(축이 둘이면 서로 다른 양의 비교로 읽힌다).
  (아래) *어떤 국면*에서 벌고 잃었나 — 슬롯대별 평균 구간수익. 여기서 유일한 마이너스가
       어느 슬롯대인지가 드러난다.

**소표본을 그림이 스스로 밝힌다.** 슬롯대별 표본은 크게 치우친다(만충이 대부분, 전환대는
각 2~5구간뿐). 평균만 크게 그리면 소표본 막대가 같은 무게로 읽히므로, **n<10 인 막대는
빗금 + 흐리게** 칠하고 막대마다 n 을 적는다. 숫자를 각주로 미루지 않고 막대 위에 둔다.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
import pandas as pd

from .report_base import ReportWriter, plt

logger = logging.getLogger(__name__)

# 슬롯 수(0~top_n) 순차 램프 — 한 색상(파랑) 밝기 단조 증가. 슬롯이 많을수록 진하다.
_RAMP = ["#e2ecf8", "#c6dcf2", "#a6c8ea", "#82b1e0", "#5c96d4", "#3a7cc4", "#2360a6", "#123f73"]
_C_POS = "#2a78d6"
_C_NEG = "#c2564a"
_C_INK = "#0b0b0b"
_C_MUTED = "#5c5b57"

_SMALL_N = 10  # 이 미만이면 소표본으로 표시(빗금 + 흐리게)


class SegmentReport(ReportWriter):
    """구간 손익을 PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    def plot_segments(self, rotations_log: Sequence[dict], top_n: int,
                      name: str = "segment_returns") -> str:
        """시기별 구간수익(색=슬롯 수) + 슬롯대별 평균(막대 위 n).

        Args:
            rotations_log: 엔진이 낸 교체 기록(`date`·`n`·`ret_pct`).
            top_n: 슬롯 수 상한(만충 판정·색 램프 스케일).

        Raises:
            ValueError: 기록이 비어 있는 경우.
        """
        if not rotations_log:
            raise ValueError("[segment_report] rotations_log 가 비어 있습니다.")
        df = pd.DataFrame([{"date": r["date"], "n": int(r["n"]), "ret": float(r["ret_pct"])}
                           for r in rotations_log])

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.2),
                                      gridspec_kw={"height_ratios": [1.35, 1]})
        self._plot_timeline(ax, df, top_n)
        self._plot_by_slot(ax2, df, top_n)
        for axis in (ax, ax2):
            axis.grid(axis="y", color=_C_MUTED, alpha=0.18, lw=0.6, zorder=0)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        return self._save(fig, name)

    # ── 위 패널 ─────────────────────────────────────────────────────
    def _plot_timeline(self, ax, df: pd.DataFrame, top_n: int) -> None:
        """교체 구간별 수익 막대 — 색이 그 구간의 슬롯 수."""
        colors = [_RAMP[min(int(n * (len(_RAMP) - 1) / max(top_n, 1)), len(_RAMP) - 1)]
                  for n in df["n"]]
        ax.bar(df["date"], df["ret"], width=22, color=colors, edgecolor="white", lw=0.5, zorder=3)
        ax.axhline(0, color=_C_INK, lw=0.8, alpha=0.6, zorder=2)
        ax.set_ylabel("구간수익(%)")
        ax.set_title("시기별 구간수익과 슬롯 수 — 충격 뒤 슬롯이 비고, 그 뒤 손실 폭이 얕아진다")

        # 최저 구간에 라벨. **슬롯 수를 함께 적는 것이 핵심**이다 — 최악의 구간이 만충
        # 상태에서 났다는 사실이 이 게이트가 예측기가 아니라 **반응기**임을 말해 준다
        # (충격을 맞은 뒤에야 슬롯이 빈다). 그림이 스스로 그 한계를 밝히게 둔다.
        worst = df.loc[df["ret"].idxmin()]
        ax.annotate(f"최저 {worst['ret']:+.1f}% ({worst['date']:%Y-%m} · {int(worst['n'])}슬롯)"
                    "\n— 첫 충격은 만충으로 맞는다(게이트는 반응기)",
                    (worst["date"], worst["ret"]), xytext=(10, 2),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=_C_MUTED, zorder=7, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82))

        # 색 범례는 슬롯 수 눈금으로 — 옅을수록 방어(적은 슬롯), 진할수록 만충.
        handles = [plt.Rectangle((0, 0), 1, 1, color=_RAMP[min(int(k * (len(_RAMP) - 1)
                                                                   / max(top_n, 1)),
                                                               len(_RAMP) - 1)])
                   for k in range(top_n + 1)]
        ax.legend(handles, [f"{k}" for k in range(top_n + 1)],
                  title=f"보유 슬롯 수 (0 = 전액 대피 · {top_n} = 만충)",
                  loc="upper left", frameon=False, fontsize=8, title_fontsize=8.5,
                  ncol=top_n + 1, handlelength=1.1, columnspacing=0.6, handletextpad=0.35)

    # ── 아래 패널 ───────────────────────────────────────────────────
    def _plot_by_slot(self, ax, df: pd.DataFrame, top_n: int) -> None:
        """슬롯대별 평균 구간수익 + 표본 수(n). 소표본은 빗금으로 구분한다."""
        g = df.groupby("n")["ret"].agg(["mean", "size"]).reindex(range(top_n + 1))
        xs = list(g.index)
        means = g["mean"].to_numpy(dtype=float)
        sizes = g["size"].fillna(0).to_numpy(dtype=int)

        for x, m, k in zip(xs, means, sizes):
            if not k or np.isnan(m):
                continue
            small = k < _SMALL_N
            ax.bar(x, m, width=0.62, zorder=3,
                   color=(_C_POS if m >= 0 else _C_NEG),
                   alpha=0.45 if small else 0.95,
                   hatch="//" if small else None,
                   edgecolor="white", lw=0.8)
            va, off = ("bottom", 4) if m >= 0 else ("top", -4)
            ax.annotate(f"{m:+.2f}%\nn={k}", (x, m), xytext=(0, off),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=8, color=_C_INK, linespacing=1.3, zorder=5)

        ax.axhline(0, color=_C_INK, lw=0.8, alpha=0.6, zorder=2)
        lo, hi = float(np.nanmin(means)), float(np.nanmax(means))
        ax.set_ylim(lo - (hi - lo) * 0.42, hi + (hi - lo) * 0.38)
        ax.set_xticks(xs)
        ax.set_xticklabels([("0\n(전액 대피)" if k == 0 else
                             (f"{k}\n(만충)" if k == top_n else str(k))) for k in xs], fontsize=9)
        ax.set_xlabel("보유 슬롯 수")
        ax.set_ylabel("평균 구간수익(%)")
        ax.text(0.5, -0.30,
                f"빗금 = 표본 n<{_SMALL_N} (방향 참고용). 전환대(4~6슬롯)는 구간이 각 2~5개뿐이고 "
                f"만충이 표본을 지배한다.",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=_C_MUTED)

    # ── 로그 ────────────────────────────────────────────────────────
    @staticmethod
    def caption_lines(rotations_log: Sequence[dict], top_n: int) -> List[str]:
        """차트와 같은 값을 콘솔로도 낸다(그림과 표가 어긋나는지 눈으로 대조하라고)."""
        df = pd.DataFrame([{"n": int(r["n"]), "ret": float(r["ret_pct"])} for r in rotations_log])
        g = df.groupby("n")["ret"].agg(["mean", "size"])
        return [f"  n={int(k)}{' 만충' if k == top_n else ''}: 평균 {row['mean']:+.2f}% "
                f"(구간 {int(row['size'])}개)" for k, row in g.iterrows()]
