"""워크포워드 오케스트레이터 — 창별 선정 → OOS 조각 이어붙이기.

흐름은 한 문장이다: **후보 곡선을 전 구간 한 번씩 만들어 두고, 창마다 학습 조각으로 후보를
고른 뒤, 그 후보의 검증 조각만 떼어 이어붙인다.** 조립은 여기 한곳에 모으고(오케스트레이터),
창 분할·선정 규칙·지표 계산은 각자의 모듈이 맡는다.

비교 기준선을 넉넉히 함께 낸다 — 워크포워드 곡선 하나만 보면 그게 좋은 건지 알 수 없다:
  · **동결(52/60·30%)** — 전 구간을 보고 고른 실제 상품. WF 가 이걸 크게 앞지르면 동결값이
    나쁜 점이었다는 뜻이고, 크게 뒤지면 선정 규칙이 잡음을 좇았다는 뜻이다. **비슷한 것이
    가장 좋은 결과**다(면이 평평하다 = plateau 주장이 표본 밖에서 성립).
  · **V1 이진60** — 원설계. Tier 2-a 가 OOS 에서도 값을 하는지.
  · **격자 평균** — 후보를 아무거나 골랐을 때의 기대값. 선정이 이것보다 못하면 선정 자체가
    해로웠다는 뜻이다(코인 던지기만 못한 규칙).
  · **오라클** — 창마다 검증 구간을 **미리 보고** 고른 상한. WF 와의 격차가 '선정의 비용'.
  · **벤치마크(TRF7030)** — 상품 대비 실물 대안.

용어: 여기서 'OOS' 는 **파라미터 선정에 쓰이지 않은 구간**을 뜻한다. 시세 자체는 전부 과거
데이터다(진짜 미래는 아무도 못 본다) — 그 한계는 `verdict` 판정문에 명시한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from irp.config import IRPConfig

from .candidates import Candidate, CandidateGrid, CandidateRunner
from .metrics import chain, curve_metrics, slice_growth, yearly_returns
from .selection import SelectionRule
from .windows import Fold, WindowScheme

logger = logging.getLogger(__name__)

# 이어붙인 OOS 곡선의 표시 라벨(리포트·판정문이 공유).
LABEL_WF = "워크포워드 OOS"
LABEL_FROZEN = "동결 52→60·30%"
LABEL_BINARY = "V1 이진60"
LABEL_GRIDMEAN = "격자 평균"
LABEL_ORACLE = "오라클(사후최적)"
LABEL_BENCH = "벤치마크"


@dataclass(frozen=True)
class FoldOutcome:
    """창 하나의 결과.

    Attributes:
        fold: 창 정의.
        picked: 학습 창만 보고 고른 후보.
        train: 선정된 후보의 **학습 창** 지표(선정이 무엇을 보고 이뤄졌는지).
        test: 선정된 후보의 **검증 창** 지표(선정 뒤에 계산).
        oracle_label / oracle_test: 검증 창을 미리 봤다면 골랐을 후보와 그 지표(상한).
        rank_in_test: 선정 후보가 검증 창 성적으로 몇 등이었나(1 = 오라클과 일치).
        n_candidates: 그 창에서 겨룬 후보 수.
        switched: 직전 창과 선정이 달라졌는지(전환 비용 부과 대상).
        runner_up: 학습 창 2등 후보 라벨(선정이 아슬아슬했는지 보기 위한 참고값).
    """
    fold: Fold
    picked: Candidate
    train: Dict[str, float]
    test: Dict[str, float]
    oracle_label: str
    oracle_test: Dict[str, float]
    rank_in_test: int
    n_candidates: int
    switched: bool
    runner_up: Optional[str] = None


@dataclass(frozen=True)
class WalkForwardResult:
    """한 선정 규칙의 워크포워드 전체 결과.

    Attributes:
        rule_name: 선정 규칙 표시명.
        scheme_name: 창 분할 방식 표시명.
        outcomes: 창별 결과(시간순).
        oos_equity: 이어붙인 OOS 자산곡선(시작 1.0, 전환비용 반영).
        curves: 같은 OOS 구간의 비교 곡선들(라벨 → 곡선). 동결·V1·격자평균·오라클·벤치마크.
        metric_key: 판정 주지표 키.
    """
    rule_name: str
    scheme_name: str
    outcomes: List[FoldOutcome]
    oos_equity: pd.Series
    curves: Dict[str, pd.Series] = field(default_factory=dict)
    metric_key: str = "calmar"

    # ── 요약 ────────────────────────────────────────────────────────
    @property
    def oos_metrics(self) -> Dict[str, float]:
        """이어붙인 OOS 곡선의 지표(낙폭은 여기서 읽어야 정확하다)."""
        return curve_metrics(self.oos_equity)

    def reference_metrics(self) -> Dict[str, Dict[str, float]]:
        """비교 곡선별 지표(같은 OOS 구간)."""
        return {label: curve_metrics(c) for label, c in self.curves.items()}

    def win_rate_vs(self, label: str, metric: Optional[str] = None) -> float:
        """창별로 선정 후보가 비교 대상을 이긴 비율(%). 비교 대상 곡선이 없으면 NaN.

        Args:
            label: 비교 곡선 라벨.
            metric: 비교 지표(None 이면 판정 주지표). MDD 처럼 **작을수록 좋은** 지표는
                부호를 뒤집어 비교한다.
        """
        curve = self.curves.get(label)
        if curve is None:
            return float("nan")
        key = metric or self.metric_key
        sign = -1.0 if key == "mdd_pct" else 1.0   # 낙폭은 0 에 가까울수록 승리
        wins = 0
        for o in self.outcomes:
            ref = curve_metrics(slice_growth(curve, o.fold.test_anchor, o.fold.test_end))
            wins += int(sign * o.test[key] > sign * ref[key])
        return wins / len(self.outcomes) * 100.0

    def win_rates_vs(self, label: str,
                     metrics: Sequence[str] = ("cagr_pct", "calmar", "sharpe", "mdd_pct")
                     ) -> Dict[str, float]:
        """여러 지표로 잰 창별 승률(%) — 판정 기준의 **보조 정보**.

        사전 등록 판정은 Calmar 하나로 한다(규율 1: 기준은 실행 전에 고정). 그런데 창 하나가
        1년뿐이면 Calmar 는 분모(그 해 낙폭)가 얕을 때 폭발한다 — 잔잔하게 오른 해의
        벤치마크 Calmar 가 6 을 넘는 식이다. 그래서 **등록 기준은 그대로 두고** 다른 지표의
        승률을 함께 낸다. 임계를 옮기는 것이 아니라 읽는 사람에게 맥락을 주는 것이다.
        """
        return {m: self.win_rate_vs(label, m) for m in metrics}

    def stability(self) -> Dict[str, float]:
        """선정의 안정성 — 창을 넘나들며 선택이 얼마나 흔들렸나.

        Returns:
            n_unique(서로 다른 선정 수) · n_switches(선정이 바뀐 횟수) ·
            mean_rank(검증 성적 기준 평균 등수) · pct_rank(백분위, 낮을수록 좋음).
        """
        picks = [o.picked.label for o in self.outcomes]
        ranks = [o.rank_in_test for o in self.outcomes]
        n_cand = self.outcomes[0].n_candidates if self.outcomes else 1
        return {
            "n_folds": float(len(self.outcomes)),
            "n_unique": float(len(set(picks))),
            "n_switches": float(sum(o.switched for o in self.outcomes)),
            "mean_rank": float(np.mean(ranks)),
            "pct_rank": float(np.mean(ranks) / n_cand * 100.0),
        }

    def fold_frame(self) -> pd.DataFrame:
        """창별 결과 표(CSV·로그 공용)."""
        rows = []
        for o in self.outcomes:
            rows.append({
                "창": o.fold.label,
                "학습": f"{o.fold.train_start:%Y-%m}~{o.fold.train_end:%Y-%m}",
                "학습개월": round(o.fold.train_months),
                "검증": f"{o.fold.test_start:%Y-%m}~{o.fold.test_end:%Y-%m}",
                "선정": o.picked.label,
                "차점": o.runner_up or "",
                "전환": "○" if o.switched else "",
                "학습CAGR%": round(o.train["cagr_pct"], 1),
                "학습Calmar": round(o.train["calmar"], 2),
                "OOS수익%": round(o.test["total_return_pct"], 1),
                "OOS_MDD%": round(o.test["mdd_pct"], 1),
                "OOS_Sharpe": round(o.test["sharpe"], 2),
                "OOS_Calmar": round(o.test["calmar"], 2),
                "오라클": o.oracle_label,
                "오라클Calmar": round(o.oracle_test["calmar"], 2),
                "검증등수": f"{o.rank_in_test}/{o.n_candidates}",
            })
        return pd.DataFrame(rows)

    def summary_lines(self) -> List[str]:
        """로그용 사람 읽는 요약(폭 지정 정렬 유지)."""
        m = self.oos_metrics
        st = self.stability()
        lines = [
            f"[{self.rule_name} · {self.scheme_name}] 이어붙인 OOS "
            f"{self.oos_equity.index[0]:%Y-%m}~{self.oos_equity.index[-1]:%Y-%m} "
            f"({int(st['n_folds'])}창)",
            f"  {'곡선':<16}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'Calmar':>8}{'최저해%':>9}",
            f"  {LABEL_WF:<16}{m['cagr_pct']:>8.1f}{m['sharpe']:>8.2f}"
            f"{m['mdd_pct']:>8.1f}{m['calmar']:>8.2f}{m['worst_year_pct']:>9.1f}",
        ]
        for label, rm in self.reference_metrics().items():
            lines.append(f"  {label:<16}{rm['cagr_pct']:>8.1f}{rm['sharpe']:>8.2f}"
                         f"{rm['mdd_pct']:>8.1f}{rm['calmar']:>8.2f}{rm['worst_year_pct']:>9.1f}")
        lines.append(
            f"  선정 안정성 · 서로 다른 선택 {int(st['n_unique'])}종 · 전환 "
            f"{int(st['n_switches'])}회 · 검증 평균 등수 {st['mean_rank']:.1f}/"
            f"{self.outcomes[0].n_candidates} (백분위 {st['pct_rank']:.0f}%, "
            f"50%=후보 한가운데)")
        # 창별 승률은 지표에 따라 크게 갈린다(1년짜리 창의 Calmar 는 분모가 얕으면 폭발).
        # 판정은 Calmar 로 하되, 읽는 사람이 오해하지 않게 다른 지표도 함께 보인다.
        for label in self.curves:
            if label.startswith(LABEL_BENCH) or label == LABEL_FROZEN:
                wr = self.win_rates_vs(label)
                txt = " · ".join(f"{k.replace('_pct', '')} {v:.0f}%" for k, v in wr.items())
                lines.append(f"  창별 승률 vs {label} — {txt}")
        return lines


class WalkForwardValidator:
    """워크포워드 검증기 — 후보 곡선·창·선정 규칙을 받아 OOS 곡선을 만든다.

    Args (생성자):
        runner: 후보 곡선을 만들어 둘 `CandidateRunner`.
        grid: 후보 격자.
        scheme: 창 분할 방식.
        rules: 선정 규칙들(각각 독립적으로 워크포워드를 돌린다).
        switch_cost: 창 경계에서 선정이 바뀔 때 물릴 비용 비율. 기본은 왕복 거래비용 1회분.
            실제 회전율은 상태 인계 없이는 알 수 없어 **보수적 대용값**이다(`candidates` 한계 참조).
        metric_key: 판정 주지표(선정 규칙과 별개로 보고·승률 계산에 쓰는 지표).
    """

    def __init__(self, runner: CandidateRunner, grid: CandidateGrid, scheme: WindowScheme,
                 rules: Sequence[SelectionRule], switch_cost: float = 0.0010,
                 metric_key: str = "calmar"):
        self.runner = runner
        self.grid = grid
        self.scheme = scheme
        self.rules = list(rules)
        self.switch_cost = float(switch_cost)
        self.metric_key = metric_key

    # ── public ──────────────────────────────────────────────────────
    def run(self, icfg: IRPConfig, start=None, end=None) -> Dict[str, WalkForwardResult]:
        """후보를 전부 돌리고, 규칙마다 워크포워드를 수행한다.

        Returns:
            규칙 표시명 → `WalkForwardResult`.
        """
        logger.info(f"후보 {len(self.grid.candidates)}개 전 구간 산출 중…")
        curves = self.runner.run_all(icfg, self.grid.candidates, start=start, end=end)
        index = next(iter(curves.values())).index
        folds = self.scheme.folds(index)
        logger.info(f"창 분할 · {self.scheme.name} · {len(folds)}창")
        for f in folds:
            logger.info(f"  {f.describe()}")

        # 창×후보 지표는 규칙마다 같으므로 한 번만 계산해 돌려 쓴다(규칙 수만큼 재계산 방지).
        train_by_fold = [self._metrics_over(curves, f.train_start, f.train_end) for f in folds]
        test_by_fold = [self._metrics_over(curves, f.test_anchor, f.test_end) for f in folds]

        out: Dict[str, WalkForwardResult] = {}
        for rule in self.rules:
            out[rule.name] = self._walk(rule, curves, folds, train_by_fold, test_by_fold)
        return out

    # ── 내부 ────────────────────────────────────────────────────────
    def _walk(self, rule: SelectionRule, curves: Dict[str, pd.Series], folds: List[Fold],
              train_by_fold: List[Dict[str, Dict[str, float]]],
              test_by_fold: List[Dict[str, Dict[str, float]]]) -> WalkForwardResult:
        """규칙 하나로 창을 굴려 OOS 곡선을 만든다."""
        outcomes: List[FoldOutcome] = []
        segments: List[pd.Series] = []
        switches: List[bool] = []
        prev: Optional[str] = None

        for f, train, test in zip(folds, train_by_fold, test_by_fold):
            picked = rule.select(train, self.grid)            # ★ 학습 창만 본다
            ranked_train = rule.ranking(train, self.grid, top=2)
            # 검증 성적 순위는 **선정 뒤에** 매긴다(선정에 쓰이지 않는다 — 사후 평가용).
            order = sorted(test.keys(), key=lambda lb: test[lb][self.metric_key], reverse=True)
            switched = prev is not None and picked.label != prev
            outcomes.append(FoldOutcome(
                fold=f, picked=picked, train=train[picked.label], test=test[picked.label],
                oracle_label=order[0], oracle_test=test[order[0]],
                rank_in_test=order.index(picked.label) + 1, n_candidates=len(order),
                switched=switched,
                runner_up=ranked_train[1][0] if len(ranked_train) > 1 else None))
            segments.append(slice_growth(curves[picked.label], f.test_anchor, f.test_end))
            switches.append(switched)
            prev = picked.label

        oos = chain(segments, switch_cost=self.switch_cost, switch_flags=switches)
        return WalkForwardResult(
            rule_name=rule.name, scheme_name=self.scheme.name, outcomes=outcomes,
            oos_equity=oos, metric_key=self.metric_key,
            curves=self._references(curves, folds, test_by_fold))

    def _references(self, curves: Dict[str, pd.Series], folds: List[Fold],
                    test_by_fold: List[Dict[str, Dict[str, float]]]) -> Dict[str, pd.Series]:
        """같은 OOS 구간의 비교 곡선들을 만든다.

        정적 기준선(동결·V1·벤치마크)은 창을 나눌 필요가 없다 — 처음부터 끝까지 같은 규칙이라
        OOS 구간을 통째로 자르면 된다. 오라클·격자평균만 창 단위로 조립한다.
        """
        anchor, last = folds[0].test_anchor, folds[-1].test_end
        refs: Dict[str, pd.Series] = {}

        frozen_label = self.frozen_label(curves)
        if frozen_label:
            refs[LABEL_FROZEN] = slice_growth(curves[frozen_label], anchor, last)
        if LABEL_BINARY in curves:
            refs[LABEL_BINARY] = slice_growth(curves[LABEL_BINARY], anchor, last)

        # 격자 평균: 후보를 무작위로 하나 골랐을 때의 기대 곡선(일간수익 단순평균).
        rets = pd.DataFrame({lb: c.loc[anchor:last].pct_change() for lb, c in curves.items()})
        refs[LABEL_GRIDMEAN] = (1.0 + rets.mean(axis=1).fillna(0.0)).cumprod().rename("growth")

        # 오라클: 창마다 검증 구간을 미리 보고 고른 상한(실현 불가능 — 선정 비용의 척도).
        segs, sw, prev = [], [], None
        for f, test in zip(folds, test_by_fold):
            best = max(test.keys(), key=lambda lb: test[lb][self.metric_key])
            segs.append(slice_growth(curves[best], f.test_anchor, f.test_end))
            sw.append(prev is not None and best != prev)
            prev = best
        refs[LABEL_ORACLE] = chain(segs, switch_cost=self.switch_cost, switch_flags=sw)

        if self.runner.benchmark is not None:
            refs[f"{LABEL_BENCH}({self.runner.benchmark_name})"] = slice_growth(
                self.runner.benchmark, anchor, last)
        return refs

    def frozen_label(self, curves: Dict[str, pd.Series]) -> Optional[str]:
        """동결 파라미터(52/60/0.3)에 해당하는 후보 라벨. 격자에 없으면 None(경고)."""
        from run_v2 import FROZEN_RAMP  # 동결값의 단일 출처 — 여기서 다시 적지 않는다
        try:
            label = self.grid.label_of(tuple(FROZEN_RAMP))
        except KeyError:
            logger.warning(f"동결값 {FROZEN_RAMP} 이 후보 격자에 없어 '동결' 기준선을 낼 수 "
                           f"없습니다 — 축 값을 동결 그리드와 맞추세요.")
            return None
        return label if label in curves else None

    @staticmethod
    def _metrics_over(curves: Dict[str, pd.Series], start, end
                      ) -> Dict[str, Dict[str, float]]:
        """모든 후보의 [start, end] 구간 지표를 계산한다."""
        return {lb: curve_metrics(slice_growth(c, start, end)) for lb, c in curves.items()}


def oos_yearly(result: WalkForwardResult) -> pd.DataFrame:
    """이어붙인 OOS 곡선의 연도별 수익 표(WF · 비교 곡선 나란히).

    '잃는 해 없음' 이라는 상품 서사가 **선정에 쓰이지 않은 구간에서도** 성립하는지 보는 표라
    별도 함수로 뺐다(결과 객체는 곡선까지만 알고, 표 조립은 바깥에서).
    """
    cols = {LABEL_WF: yearly_returns(result.oos_equity)}
    cols.update({lb: yearly_returns(c) for lb, c in result.curves.items()})
    return pd.DataFrame(cols).round(1)
