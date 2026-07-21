"""실효 노출률 산출물 — CSV + 차트(제안서 삽입용).

구 '슬롯 미달' 차트의 후계다. 서사(위기에는 셔터가 내려간다)는 그대로 계승하되, 동결 V2 의
정체성인 **계단 → 경사로**가 그림 자체에서 읽히게 한다.

그림의 장치: 가로 격자를 슬롯 경계(슬롯 1개 = 포트폴리오 10%)에 둔다. V1 은 슬롯을 채우거나
말거나라 이 선에 **정확히 얹히고**(계단), V2 는 부분 충전이라 선 **사이에 뜬다**(경사로).
같은 축·같은 단위에서 두 세대의 차이가 눈으로 구분된다.
"""
from __future__ import annotations

from typing import Optional

from .exposure import ExposureResult
from .report_base import ReportWriter, plt  # plt 는 폰트 적용 뒤의 pyplot(베이스에서 준비)

# 색 — 팔레트 slot 1(파랑)은 제안서가 파는 상품(동결 V2)에 준다. V1 은 '각주·비교용'이므로
# 동급 시리즈 색을 주지 않고 후퇴색(먹색)으로 둔다. 색만으로 구분되지 않도록 선 스타일
# (실선+면 vs 점선)로 이중 부호화한다 — 흑백 인쇄·색각 이상에서도 갈린다.
_C_V2 = "#2a78d6"
_C_V1 = "#52514e"
_C_INK = "#0b0b0b"
_C_GRID = "#8a8984"


class ExposureReport(ReportWriter):
    """실효 노출률 결과를 CSV·PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    # ── CSV ─────────────────────────────────────────────────────────
    def write_monthly(self, v2: ExposureResult, v1: Optional[ExposureResult] = None,
                      name: str = "exposure_monthly") -> str:
        """체크 시점별 노출 상세. V1 은 이진 환산 참고치로 열을 덧붙인다."""
        df = v2.to_frame()
        if v1 is not None:
            df["V1참고_포트폴리오노출%"] = (v1.portfolio_exposure * 100).round(2)
            df["V1참고_소비슬롯"] = v1.slots_used
        return self._write_csv(df, name, index=True)  # 날짜 인덱스를 열로 보존

    # ── 차트 ────────────────────────────────────────────────────────
    def plot_exposure(self, v2: ExposureResult, v1: Optional[ExposureResult] = None,
                      v2_label: str = "동결 V2", v1_label: str = "V1 기준",
                      name: str = "exposure_monthly") -> str:
        """월별 실효 노출률 — V1 계단 vs V2 경사로."""
        fig, ax = plt.subplots(figsize=(11, 4.8))
        full_pct = v2.sleeve_weight * 100.0
        slot_pct = full_pct / v2.top_n

        # 슬롯 경계 격자 — 이 그림의 논지가 여기서 보인다(V1 은 선 위, V2 는 선 사이).
        for k in range(1, v2.top_n + 1):
            ax.axhline(k * slot_pct, color=_C_GRID, lw=0.6, alpha=0.35, zorder=1)

        if v1 is not None:
            y1 = v1.portfolio_exposure * 100.0
            ax.step(y1.index, y1.to_numpy(), where="post", color=_C_V1, lw=1.3, ls="--",
                    label=v1_label, zorder=2)

        y2 = v2.portfolio_exposure * 100.0
        # 체크에서 정한 노출은 다음 체크까지 유지된다 → 시간축에서 계단(steps-post)이 맞다.
        ax.fill_between(y2.index, y2.to_numpy(), step="post", color=_C_V2, alpha=0.15,
                        lw=0, zorder=2)
        ax.step(y2.index, y2.to_numpy(), where="post", color=_C_V2, lw=2.0,
                label=v2_label, zorder=3)

        # 그림 읽는 법은 주석 한 줄로. 범례를 축 안에 두면 만충 근처 데이터를 가리므로 밖으로 뺀다.
        ax.text(y2.index[0], full_pct * 1.06,
                f"만충 = 사테라이트 {full_pct:.0f}%  ·  가로선 = 슬롯 경계(1개 = {slot_pct:.0f}%)",
                fontsize=8.5, color=_C_INK, va="bottom")
        ax.set_ylim(0, full_pct * 1.20)
        ax.set_ylabel("포트폴리오 위험자산 노출(%)")
        ax.set_title("월별 실효 노출률 — 이진 계단에서 점수 비례 경사로")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
                  frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return self._save(fig, name)
