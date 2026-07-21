"""적립식 분석 산출물 — CSV + 차트(제안서 §8~9 삽입용).

숫자를 화면에 뿌리는 것과 문서에 넣을 그림을 만드는 것은 다른 책임이라 여기 모은다.
한글 라벨이 깨지면 제안서에 그대로 실리므로 폰트 설정은 이 모듈에서 한 번만 한다.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .dca import DCAResult
from .report_base import ReportWriter, plt  # plt 는 폰트 적용 뒤의 pyplot(베이스에서 준비)
from .rolling import HorizonStats


class DCAReport(ReportWriter):
    """적립식 분석 결과를 CSV·PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    # ── CSV ─────────────────────────────────────────────────────────
    def write_summary(self, results: Dict[str, List[DCAResult]], name: str = "dca_summary") -> str:
        """전략별·납입계획별 요약표(총 납입액 대비 평가액 + MWR/TWR)."""
        rows = []
        for strat, rs in results.items():
            for r in rs:
                rows.append({
                    "전략": strat, "납입계획": r.plan_name,
                    "시작": f"{r.start:%Y-%m-%d}", "종료": f"{r.end:%Y-%m-%d}",
                    "납입횟수": r.n_payments,
                    "총납입액": round(r.contributed),
                    "최종평가액": round(r.final_value),
                    "손익": round(r.profit),
                    "납입액대비손익률%": round(r.profit_pct, 2),
                    "MWR연율%": None if r.mwr_pct is None else round(r.mwr_pct, 2),
                    "TWR연율%": round(r.twr_pct, 2),
                })
        return self._write_csv(pd.DataFrame(rows), name)

    def write_rolling(self, stats: Dict[str, List[HorizonStats]], name: str = "dca_rolling") -> str:
        """보유기간별 롤링 통계(손실 확률·중앙값·최악)."""
        rows = []
        for strat, ss in stats.items():
            for s in ss:
                rows.append({
                    "전략": strat, "납입계획": s.plan_name,
                    "보유개월": s.horizon_months, "표본창수": s.n_windows,
                    "손실확률%": round(s.loss_prob_pct, 1),
                    "손익률중앙값%": round(s.median_pct, 2),
                    "최악%": round(s.worst_pct, 2), "최선%": round(s.best_pct, 2),
                    "MWR중앙값%": round(s.median_mwr_pct, 2),
                })
        return self._write_csv(pd.DataFrame(rows), name)

    # ── 차트 ────────────────────────────────────────────────────────
    def plot_loss_curve(self, stats: Dict[str, List[HorizonStats]],
                        name: str = "dca_loss_curve") -> str:
        """보유기간별 손실 확률 곡선 — "몇 년 들면 잃지 않나" 에 답하는 그림."""
        fig, ax = plt.subplots(figsize=(9, 5))
        for strat, ss in stats.items():
            for plan in sorted({s.plan_name for s in ss}):
                pts = sorted([s for s in ss if s.plan_name == plan], key=lambda s: s.horizon_months)
                ax.plot([s.horizon_months for s in pts], [s.loss_prob_pct for s in pts],
                        marker="o", label=f"{strat} · {plan}")
        ax.set_xlabel("보유기간(개월)")
        ax.set_ylabel("손실 확률(%)  — 평가액 < 총 납입액")
        ax.set_title("보유기간별 손실 확률 (모든 시작 월 롤링)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        return self._save(fig, name)

    def plot_distribution(self, dists: Dict[str, pd.Series], horizon_months: int,
                          name: str = "dca_distribution") -> str:
        """시작 월별 손익률 — "언제 시작했느냐" 의 영향을 보여주는 그림."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, ser in dists.items():
            ax.plot(ser.index, ser.to_numpy(), marker="o", ms=3, label=label)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("가입(시작) 월")
        ax.set_ylabel("총 납입액 대비 손익률(%)")
        ax.set_title(f"시작 시점별 성과 분포 (보유 {horizon_months}개월)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        return self._save(fig, name)

    def plot_growth(self, curves: Dict[str, pd.DataFrame], name: str = "dca_growth") -> str:
        """납입 누계 대 평가액 추이 — 제안서 표지급 그림(원금선 위에 자산이 쌓이는 모습)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, df in curves.items():
            ax.plot(df.index, df["value"] / 1e4, label=f"{label} 평가액")
        first = next(iter(curves.values()))
        ax.plot(first.index, first["paid"] / 1e4, color="black", ls="--", lw=1.2, label="납입 누계(원금)")
        ax.set_xlabel("연도")
        ax.set_ylabel("금액(만원)")
        ax.set_title("월 적립 시 납입 누계 대비 평가액 추이")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        return self._save(fig, name)
