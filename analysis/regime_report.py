"""국면 분해 산출물 — 국면 음영 자산곡선 차트(발표 슬라이드용).

**이 그림이 말하는 것.** §6.3 표(국면별 수익·MDD·노출)를 한 장으로 압축한다. 배경 음영이
하락·횡보 국면을 표시하고, 그 구간에서 세 곡선의 간격이 벌어지는 모습이 곧 '방어'다. 아래
패널의 노출이 같은 구간에서 내려앉는 것이 그 **인과**다 — 위에서 덜 잃은 이유가 아래에 있다.
표로는 두 사실을 나란히 놓을 수만 있지만, 같은 시간축에 겹치면 인과가 눈으로 읽힌다.

**음영 규약.** 상승 국면(전체의 57.5%)은 칠하지 않는다. 가장 흔한 상태를 배경(=무채)으로 두면
잉크가 예외 구간에만 남아 하락·횡보가 도드라진다. 하락은 경고색, 횡보는 중립 회색.

**색 규약(`exposure_report.py` 와 공유).** 파랑은 제안서가 파는 상품(동결 V2)의 자리다.
벤치마크는 별개 색상 슬롯(주황)을 받고, V1 원설계는 '비교용 각주'이므로 동급 시리즈 색을 주지
않고 후퇴색(먹색 점선)으로 둔다 — 색각 이상·흑백 인쇄에서도 갈리도록 선 스타일로 이중
부호화한다. 주 색상 2종(파랑·주황)은 팔레트 검사 6항목을 전부 통과하며(CVD ΔE 26.3),
먹색은 채도 하한 검사에서 의도적으로 벗어난 회색 = 시리즈가 아니라 배경 참조선이라는 신호다.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional, Tuple

import pandas as pd

from .regime import DOWN, FLAT, UP
from .report_base import ReportWriter, plt  # plt 는 폰트 적용 뒤의 pyplot(베이스에서 준비)

logger = logging.getLogger(__name__)

# 시리즈 색 — 파랑=상품, 주황=벤치마크(별개 슬롯), 먹색 점선=비교용 원설계.
_C_V2 = "#2a78d6"
_C_V1 = "#52514e"
_C_BENCH = "#c2681a"
_C_INK = "#0b0b0b"
_C_MUTED = "#5c5b57"

# 국면 음영 — 상승은 칠하지 않는다(가장 흔한 상태 = 배경).
_BAND = {DOWN: ("#c2564a", 0.11), FLAT: ("#8a8984", 0.09)}


def _blocks(labels: pd.Series) -> list:
    """연속 같은 라벨 구간을 (시작, 끝, 라벨) 로 묶는다.

    끝은 **다음 블록의 시작**으로 잡아야 음영 사이에 흰 틈이 생기지 않는다(마지막 블록만
    자기 끝을 쓴다). 라벨이 1~3일짜리로 잘게 쪼개지는 구간이 있어 틈이 눈에 띈다.
    """
    grp = (labels != labels.shift()).cumsum()
    segs = [(s.index[0], s.index[-1], s.iloc[0]) for _, s in labels.groupby(grp)]
    out = []
    for i, (t0, t1, lab) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else t1
        out.append((t0, end, lab))
    return out


class RegimeReport(ReportWriter):
    """국면 분해 결과를 PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    def plot_regime_equity(self, curves: Mapping[str, pd.Series], labels: pd.Series,
                           exposure: Optional[pd.Series] = None,
                           sleeve_weight: float = 0.70,
                           highlight: Optional[str] = None,
                           name: str = "regime_equity") -> str:
        """국면 음영 자산곡선(+ 하단 노출 패널).

        Args:
            curves: {표시명: 자산곡선}. 첫 항목(또는 `highlight`)이 강조 대상(상품)이다.
            labels: 일별 국면 라벨(`regime.classify_regime()`).
            exposure: 체크 시점별 포트폴리오 노출(0~1). 주면 하단 패널을 그린다.
            sleeve_weight: 만충 기준선(사테라이트 비중). 하단 패널 눈금 해설용.
            highlight: 강조할 곡선 이름. None 이면 첫 곡선.
        """
        names = list(curves)
        hi = highlight or names[0]
        # 곡선이 3종이면 색·선스타일을 이중 부호화해 배정한다(상품 → 벤치 → 각주 순).
        style = self._styles(names, hi)

        if exposure is not None:
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
        else:
            fig, ax = plt.subplots(figsize=(11, 5.0))
            ax2 = None

        for axis in (ax, ax2):
            if axis is not None:
                self._shade(axis, labels)

        finals = {}
        for label in names:
            color, ls, lw, z = style[label]
            y = (curves[label] - 1.0) * 100.0
            ax.plot(y.index, y.to_numpy(), color=color, ls=ls, lw=lw, zorder=z, label=label)
            finals[label] = (float(y.iloc[-1]), color, z)

        ax.axhline(0, color=_C_MUTED, lw=0.7, alpha=0.5, zorder=1)
        ax.set_ylabel("누적 수익률(%)")
        ax.set_title("국면별 자산곡선 — 하락·횡보 구간에서 벌어지는 간격이 방어다")
        ax.margins(x=0.01)

        # 직접 라벨 — 세 곡선의 최종값이 서로 5%p 안쪽이라 그대로 찍으면 글자가 겹친다.
        # 축 안 오른쪽에 최소 간격을 강제해 흩뿌린다(값 자체는 라벨에 그대로 적는다).
        self._end_labels(ax, finals, curves[names[0]].index[-1])

        # 범례는 하나로 모은다 — 시리즈 3 + 음영 2. 좌상단은 이 그림에서 비는 자리다.
        bands = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=a, lw=0)
                 for c, a in (_BAND[DOWN], _BAND[FLAT])]
        series = [plt.Line2D([], [], color=style[n][0], ls=style[n][1], lw=style[n][2])
                  for n in names]
        leg = ax.legend(series + bands, names + [f"{DOWN} 국면(음영)", f"{FLAT} 국면(음영)"],
                        loc="upper left", frameon=False, fontsize=8.5, ncol=2,
                        handlelength=1.8, columnspacing=1.4, labelspacing=0.35,
                        title=f"무음영 = {UP} 국면(전체의 {(labels == UP).mean() * 100:.1f}%)")
        leg.get_title().set_fontsize(8.5)
        leg.get_title().set_color(_C_MUTED)
        leg._legend_box.align = "left"  # 제목을 항목과 같은 왼쪽 기준선에 맞춘다

        if ax2 is not None:
            full = sleeve_weight * 100.0
            y = exposure * 100.0
            # 체크에서 정한 노출은 다음 체크까지 유지된다 → 계단(steps-post)이 맞다.
            ax2.fill_between(y.index, y.to_numpy(), step="post", color=_C_V2, alpha=0.18, lw=0)
            ax2.step(y.index, y.to_numpy(), where="post", color=_C_V2, lw=1.4)
            ax2.axhline(full, color=_C_MUTED, lw=0.7, ls=":", alpha=0.7)
            ax2.text(y.index[0], full, f" 만충 {full:.0f}%", fontsize=8, color=_C_MUTED,
                     va="bottom")
            ax2.set_ylim(0, full * 1.18)
            ax2.set_ylabel("노출(%)")
            ax2.text(0.995, 0.08, "실효 노출(만충 대비 충전율 × 사테라이트 70%)",
                     transform=ax2.transAxes, color=_C_V2, fontsize=8.5, ha="right")

        for axis in (ax, ax2):
            if axis is None:
                continue
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        return self._save(fig, name)

    # ── 내부 ────────────────────────────────────────────────────────
    @staticmethod
    def _end_labels(ax, finals: dict, x_end) -> None:
        """곡선 끝 직접 라벨을 겹치지 않게 흩뿌린다.

        값 순으로 세우고 축 높이의 6% 를 최소 간격으로 강제한다. 라벨 위치는 밀려도 **적는
        숫자는 실제 최종값** 그대로다(위치를 보고 값을 읽는 그림이 아니라, 곡선 끝을 짚어
        주는 이름표이므로).
        """
        lo, hi_ = ax.get_ylim()
        gap = (hi_ - lo) * 0.06
        placed = []
        for label, (val, color, z) in sorted(finals.items(), key=lambda kv: -kv[1][0]):
            y = val if not placed else min(val, placed[-1] - gap)
            placed.append(y)
            ax.annotate(f"{label} {val:+.0f}%", (x_end, y), xytext=(-6, 0),
                        textcoords="offset points", color=color, fontsize=8.5,
                        va="center", ha="right", zorder=z + 10,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72))

    @staticmethod
    def _styles(names, highlight: str) -> dict:
        """곡선별 (색, 선스타일, 두께, zorder). 강조 대상만 굵은 실선 파랑."""
        rest = [n for n in names if n != highlight]
        out = {highlight: (_C_V2, "-", 2.1, 5)}
        # 벤치마크(첫 비강조)는 색상 슬롯을, 그 다음(원설계 등)은 후퇴색을 준다.
        for i, n in enumerate(rest):
            out[n] = (_C_BENCH, "-", 1.5, 4) if i == 0 else (_C_V1, "--", 1.3, 3)
        return out

    @staticmethod
    def _shade(ax, labels: pd.Series) -> None:
        """하락·횡보 국면에 배경 음영을 깐다(상승은 무음영)."""
        for t0, t1, lab in _blocks(labels):
            band: Optional[Tuple[str, float]] = _BAND.get(lab)
            if band is None:
                continue
            color, alpha = band
            ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, zorder=0)
