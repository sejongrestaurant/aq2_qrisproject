"""워크포워드 산출물 — CSV + 차트(제안서·발표 삽입용).

세 장이 각각 다른 질문에 답한다(한 장에 다 넣으면 아무 질문에도 답하지 못한다):

  ① `walkforward_curves`  — "표본 밖에서 곡선이 어떻게 생겼나." 이어붙인 OOS 곡선과 정적
     기준선(동결·벤치마크)을 같은 축에. 창 경계를 세로선으로 그어 **선정이 갱신된 지점**을
     보인다. 이 그림의 논지는 '동결 곡선과 겹쳐 보인다' 여야 한다 — 겹칠수록 면이 평평하다.
  ② `walkforward_folds`   — "창마다 이겼나 졌나." 창별 OOS Calmar 를 WF·동결·벤치마크로
     묶어 비교. 평균 하나로 숨는 것을 막는다(한 창의 대박이 평균을 들어 올리는 경우).
  ③ `walkforward_spread`  — "고르는 것이 중요하긴 한가." 창마다 **후보 46개 전부**의 OOS
     Calmar 를 뿌리고 선정·오라클을 얹는다. 점 구름이 좁으면 선택이 성과를 좌우하지 않는다는
     뜻이고, 그게 곧 plateau 의 그림이다.

색: 상품(동결)은 이 저장소가 이미 쓰는 파랑(`analysis/exposure_report.py`)을 그대로 유지하고,
이 리포트의 주인공인 워크포워드에 주황을 준다. 벤치마크·후보 구름은 정체성을 다투는 계열이
아니므로 채도 없는 후퇴색으로 두고, 선 스타일(점선)·직접 라벨로 이중 부호화한다(흑백 인쇄·
색각 이상 대비). 세 계열 팔레트는 검증 스크립트 6종 검사 전부 통과.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.report_base import ReportWriter, plt

from .metrics import curve_metrics, slice_growth
from .verdict import Verdict
from .walkforward import (LABEL_BENCH, LABEL_FROZEN, LABEL_ORACLE, LABEL_WF,
                          WalkForwardResult, oos_yearly)

logger = logging.getLogger(__name__)

# 계열 색 — 파랑은 상품(동결), 주황은 워크포워드, 초록은 오라클(상한).
_C_WF = "#d1670a"
_C_FROZEN = "#2a78d6"
_C_ORACLE = "#1a8a6b"
# 후퇴색: 벤치마크·후보 구름·격자. 계열 색을 주지 않아 '비교용' 임을 색으로도 말한다.
_C_BENCH = "#52514e"
_C_CLOUD = "#8a8984"
_C_INK = "#0b0b0b"

# 후보 분포를 '창 중앙값 대비 배수' 로 그릴 때, 중앙값이 이보다 작으면 나눗셈이 불안정해진다
# (0 근처에서 배수가 폭발하고 음수면 대소가 뒤집힌다) → 원값 그리기로 되돌린다.
_MIN_MEDIAN_FOR_RATIO = 0.05


class WalkForwardReport(ReportWriter):
    """워크포워드 결과를 CSV·PNG 로 떨군다(저장 공통부는 `ReportWriter`)."""

    # ── CSV ─────────────────────────────────────────────────────────
    def write_folds(self, result: WalkForwardResult, name: str = "walkforward_folds") -> str:
        """창별 선정·학습 성적·OOS 성적 표."""
        return self._write_csv(result.fold_frame(), name)

    def write_summary(self, results: Dict[str, WalkForwardResult],
                      is_metrics: Optional[Dict[str, float]] = None,
                      name: str = "walkforward_summary") -> str:
        """규칙별 이어붙인 OOS 지표 + 비교 곡선 지표를 한 표로.

        Args:
            is_metrics: 인샘플 기준(동결 전 구간) 지표. 있으면 맨 위에 참고 행으로 넣어
                열화 폭을 표에서 바로 읽게 한다.
        """
        rows: List[dict] = []
        if is_metrics is not None:
            rows.append(self._metric_row("(참고) 인샘플 동결 전구간", "—", is_metrics))
        seen_refs = False
        for rule_name, res in results.items():
            rows.append(self._metric_row(LABEL_WF, rule_name, res.oos_metrics))
            if not seen_refs:            # 비교 곡선은 규칙과 무관 — 한 번만 싣는다
                for label, m in res.reference_metrics().items():
                    rows.append(self._metric_row(label, "—", m))
                seen_refs = True
        return self._write_csv(pd.DataFrame(rows), name)

    def write_yearly(self, result: WalkForwardResult,
                     name: str = "walkforward_yearly") -> str:
        """OOS 구간 연도별 수익 표(WF vs 비교 곡선)."""
        df = oos_yearly(result)
        df.index.name = "연도"
        return self._write_csv(df, name, index=True)

    def write_curves(self, result: WalkForwardResult,
                     name: str = "walkforward_oos") -> str:
        """이어붙인 OOS 일간 곡선(재현·추가 분석용 원자료)."""
        df = pd.DataFrame({LABEL_WF: result.oos_equity, **result.curves})
        df.index.name = "date"
        return self._write_csv(df.round(6), name, index=True)

    def write_verdicts(self, verdicts: Dict[str, Verdict],
                       name: str = "walkforward_verdict") -> str:
        """규칙별·기준별 판정 결과(사전 등록 임계값 포함)."""
        rows = [{
            "규칙": rule, "기준": c.name, "질문": c.question,
            "실측": round(c.value, 3), "임계": round(c.threshold, 3),
            "판정": "통과" if c.passed else "미달", "상세": c.detail,
        } for rule, v in verdicts.items() for c in v.checks]
        return self._write_csv(pd.DataFrame(rows), name)

    # ── 차트 ────────────────────────────────────────────────────────
    def plot_curves(self, result: WalkForwardResult,
                    name: str = "walkforward_curves") -> str:
        """① 이어붙인 OOS 곡선 vs 정적 기준선(창 경계 표시)."""
        fig, ax = plt.subplots(figsize=(11, 5.0))
        wf = result.oos_equity

        bench_label = next((lb for lb in result.curves if lb.startswith(LABEL_BENCH)), None)
        if bench_label:
            b = result.curves[bench_label]
            ax.plot(b.index, b.to_numpy(), color=_C_BENCH, lw=1.4, ls=":", zorder=2,
                    label=bench_label)
        if LABEL_FROZEN in result.curves:
            f = result.curves[LABEL_FROZEN]
            ax.plot(f.index, f.to_numpy(), color=_C_FROZEN, lw=2.0, zorder=3, label=LABEL_FROZEN)
        ax.plot(wf.index, wf.to_numpy(), color=_C_WF, lw=2.4, zorder=4, label=LABEL_WF)

        # 창 경계 — 선정이 갱신된 지점. 눈금이 아니라 배경이므로 아주 옅게.
        for o in result.outcomes[1:]:
            ax.axvline(o.fold.test_start, color=_C_CLOUD, lw=0.8, alpha=0.45, zorder=1)
        for o in result.outcomes:
            ax.annotate(o.picked.label, xy=(o.fold.test_start, ax.get_ylim()[0]),
                        xytext=(3, 6), textcoords="offset points", fontsize=7.5,
                        color=_C_CLOUD, rotation=90, va="bottom", zorder=1)

        ax.axhline(1.0, color=_C_CLOUD, lw=0.6, alpha=0.5, zorder=1)
        ax.set_title(f"표본 밖(OOS) 자산곡선 — {result.rule_name} · {result.scheme_name}",
                     fontsize=12, color=_C_INK)
        ax.set_ylabel("성장배수(구간 시작 = 1.0)", fontsize=9)
        self._despine(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        return self._save(fig, name)

    def plot_folds(self, result: WalkForwardResult, name: str = "walkforward_folds") -> str:
        """② 창별 OOS Calmar — WF vs 동결 vs 벤치마크."""
        labels = [o.fold.label for o in result.outcomes]
        wf = [o.test["calmar"] for o in result.outcomes]
        series = [(LABEL_WF, wf, _C_WF)]
        for ref_label, color in ((LABEL_FROZEN, _C_FROZEN), (LABEL_BENCH, _C_BENCH)):
            hit = next((lb for lb in result.curves if lb.startswith(ref_label)), None)
            if hit:
                series.append((hit, self._fold_metric(result, hit, "calmar"), color))

        fig, ax = plt.subplots(figsize=(11, 4.6))
        x = np.arange(len(labels))
        # 막대 사이 2px 상당의 여백을 남긴다(폭 0.26 × 3계열 = 0.78 < 1.0).
        width = 0.26
        for k, (lb, vals, color) in enumerate(series):
            off = (k - (len(series) - 1) / 2) * width
            ax.bar(x + off, vals, width * 0.92, color=color, label=lb, zorder=3)
        for k, v in enumerate(wf):   # 주인공 계열만 직접 라벨(모든 점에 숫자 금지)
            off = (0 - (len(series) - 1) / 2) * width
            ax.annotate(f"{v:.2f}", xy=(x[k] + off, v), xytext=(0, 3 if v >= 0 else -11),
                        textcoords="offset points", ha="center", fontsize=8, color=_C_INK)

        ax.axhline(0.0, color=_C_INK, lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("검증 구간 Calmar", fontsize=9)
        ax.set_title(f"창별 표본 밖 성적 — {result.rule_name}", fontsize=12, color=_C_INK)
        self._despine(ax)
        ax.legend(loc="best", frameon=False, fontsize=9)
        return self._save(fig, name)

    def plot_spread(self, result: WalkForwardResult, all_curves: Dict[str, pd.Series],
                    name: str = "walkforward_spread") -> str:
        """③ 창별 후보 전체의 OOS 성적 분포 + 선정·오라클 위치.

        점 구름의 세로 폭이 곧 '선택이 성과를 얼마나 가르는가' 다. 좁으면 plateau.

        **창 중앙값 대비 배수**로 그린다. 창 Calmar 의 절대 수준은 창마다 몇 배씩 다르고
        (분모인 그 해 낙폭이 얕으면 폭발한다), 원값 그대로 그리면 한 창의 큰 값이 축을 잡아
        먹어 나머지 창의 구름이 납작해진다 — 정작 이 그림이 보여야 할 '구름의 폭' 이 안 보인다.
        중앙값으로 나누면 1.0 = 그 창의 평범한 후보가 되어 창끼리 폭을 견줄 수 있다.
        (중앙값이 0 근처인 창이 하나라도 있으면 나눗셈이 뒤집히므로 전 창을 원값으로 되돌린다.)
        """
        by_fold = [[curve_metrics(slice_growth(c, o.fold.test_anchor, o.fold.test_end))["calmar"]
                    for c in all_curves.values()] for o in result.outcomes]
        medians = [float(np.median(v)) for v in by_fold]
        normalize = all(m > _MIN_MEDIAN_FOR_RATIO for m in medians)
        if not normalize:
            logger.warning("창 중앙값이 0 에 가까운 창이 있어 후보 분포를 원값(Calmar)으로 "
                           "그립니다 — 창끼리 폭 비교는 하지 마세요.")
        scale = medians if normalize else [1.0] * len(medians)

        fig, ax = plt.subplots(figsize=(11, 4.6))
        x = np.arange(len(result.outcomes))
        for k, o in enumerate(result.outcomes):
            vals = np.array(by_fold[k]) / scale[k]
            jitter = (np.random.default_rng(k).random(len(vals)) - 0.5) * 0.22
            ax.scatter(x[k] + jitter, vals, s=16, color=_C_CLOUD, alpha=0.55, lw=0, zorder=2,
                       label="후보 전체" if k == 0 else None)
            ax.scatter([x[k]], [o.oracle_test["calmar"] / scale[k]], s=90, marker="_", lw=2.2,
                       color=_C_ORACLE, zorder=4, label=LABEL_ORACLE if k == 0 else None)
            ax.scatter([x[k]], [o.test["calmar"] / scale[k]], s=72, marker="D", color=_C_WF,
                       edgecolor="white", lw=1.2, zorder=5,
                       label="선정 후보" if k == 0 else None)

        if normalize:
            ax.axhline(1.0, color=_C_INK, lw=0.8, ls="--", alpha=0.6, zorder=1)
            ylab = "창 중앙값 대비 배수 (1.0 = 그 창의 평범한 후보)"
            ticks = [f"{o.fold.label}\n중앙값 {m:.2f}" for o, m in zip(result.outcomes, medians)]
        else:
            ax.axhline(0.0, color=_C_INK, lw=0.8, zorder=1)
            ylab = "검증 구간 Calmar(원값)"
            ticks = [o.fold.label for o in result.outcomes]

        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(f"창별 후보 분포 — 선택이 성과를 얼마나 가르나 ({result.rule_name})",
                     fontsize=12, color=_C_INK)
        self._despine(ax)
        ax.legend(loc="best", frameon=False, fontsize=9)
        return self._save(fig, name)

    # ── 내부 ────────────────────────────────────────────────────────
    @staticmethod
    def _fold_metric(result: WalkForwardResult, curve_label: str, key: str) -> List[float]:
        """비교 곡선의 창별 지표(WF 와 같은 창으로 잘라서 잰다)."""
        curve = result.curves[curve_label]
        return [curve_metrics(slice_growth(curve, o.fold.test_anchor, o.fold.test_end))[key]
                for o in result.outcomes]

    @staticmethod
    def _metric_row(curve: str, rule: str, m: Dict[str, float]) -> dict:
        """요약표 한 행(지표 이름·반올림을 한곳에서)."""
        return {
            "곡선": curve, "선정규칙": rule,
            "CAGR%": round(m["cagr_pct"], 2), "Sharpe": round(m["sharpe"], 2),
            "Sortino": round(m["sortino"], 2), "MDD%": round(m["mdd_pct"], 2),
            "Calmar": round(m["calmar"], 3), "Ulcer": round(m["ulcer"], 2),
            "최저해%": round(m["worst_year_pct"], 1),
            "총수익%": round(m["total_return_pct"], 1),
        }

    @staticmethod
    def _despine(ax) -> None:
        """축·격자를 후퇴시킨다(데이터가 가장 진하게 남도록)."""
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_C_CLOUD)
            ax.spines[side].set_linewidth(0.8)
        ax.grid(axis="y", color=_C_CLOUD, lw=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_C_INK, labelsize=9, length=0)
