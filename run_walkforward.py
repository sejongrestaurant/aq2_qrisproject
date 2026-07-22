"""OOS 워크포워드 검증 러너 — "동결 수치가 커브피팅인가" 에 답하는 산출물.

기준선 수치(전체 CAGR 13.4 · Calmar 1.05)는 2020-01~2026-06 **전 구간을 보고** 고른
파라미터의 성적이다. 이 러너는 같은 선정 절차를 **창마다 그때까지의 데이터만으로** 되풀이하고,
그 선택을 **아직 보지 않은 다음 구간**에 적용해 이어붙인다. 그래서 나오는 곡선은 파라미터
선정에 한 번도 쓰이지 않은 구간만으로 이뤄진다.

**이 러너는 상품을 바꾸지 않는다.** 2026-07-15 실험 동결이 발효 중이고, 워크포워드가 무엇을
고르든 동결값(52/60/0.3)은 그대로다. 여기서 고르는 후보는 '그 시점에 이 규칙을 썼다면 무엇을
골랐을까' 의 재현이지 채택 후보가 아니다(규율 위반 아님 — 산출물은 검증 기록이다).

실행:
    python run_walkforward.py                          # 앵커드 24개월 학습 · 12개월 검증
    python run_walkforward.py --scheme rolling         # 롤링 고정폭 학습
    python run_walkforward.py --test-months 6          # 검증 창을 반년으로(창 수 ↑)
    python run_walkforward.py --grid quick             # 축소 격자(구조 점검용)

산출물(`reports/`):
    walkforward_summary.csv   규칙별 이어붙인 OOS 지표 + 비교 곡선(동결·V1·격자평균·오라클·벤치)
    walkforward_folds.csv     창별 선정·학습 성적·OOS 성적·오라클 대비 등수
    walkforward_yearly.csv    OOS 구간 연도별 수익('잃는 해 없음' 을 표본 밖에서 재확인)
    walkforward_verdict.csv   사전 등록 기준별 통과/미달
    walkforward_oos.csv       이어붙인 OOS 일간 곡선(원자료)
    walkforward_curves.png    OOS 곡선 vs 동결 vs 벤치마크(창 경계 표시)
    walkforward_folds.png     창별 OOS Calmar 비교
    walkforward_spread.png    창별 후보 분포 — 선택이 성과를 얼마나 가르나
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict

from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from validation import (AnchoredWindows, CandidateGrid, CandidateRunner, MemoIndicator,
                        MemoLoader, RollingWindows, WalkForwardValidator, curve_metrics)
from validation.candidates import AXES_FULL, AXES_QUICK
from validation.report import WalkForwardReport
from validation.selection import default_rules
from validation.verdict import compare_rules, judge
from validation.walkforward import LABEL_FROZEN

logger = logging.getLogger("run_walkforward")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IRP 전략 OOS 워크포워드 검증")
    ap.add_argument("--start", default=None, help="시작일 override(YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="종료일 override(YYYY-MM-DD).")
    ap.add_argument("--scheme", choices=("anchored", "rolling"), default="anchored",
                    help="창 분할 — anchored(확장 학습, 기본) · rolling(고정폭 학습).")
    ap.add_argument("--min-train-months", type=int, default=24,
                    help="앵커드 첫 학습 창 길이(개월). 지표 워밍업 252봉보다 넉넉해야 한다.")
    ap.add_argument("--train-months", type=int, default=36,
                    help="롤링 학습 창 길이(개월).")
    ap.add_argument("--test-months", type=int, default=12,
                    help="검증 창 길이(개월). 창을 미는 간격도 같은 값으로 고정된다.")
    ap.add_argument("--min-test-months", type=int, default=6,
                    help="마지막 자투리 검증 창을 버리는 하한(개월).")
    ap.add_argument("--grid", choices=("full", "quick"), default="full",
                    help="후보 격자 — full(동결 근거와 같은 3축 45조합) · quick(축소, 점검용).")
    ap.add_argument("--switch-cost", type=float, default=None,
                    help="창 경계에서 선정이 바뀔 때 물릴 비용 비율(기본 config 의 왕복 거래비용).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="유니버스 종목이 빠져도 진행(기본은 중단). 빠진 채 나온 결과는 "
                         "설정과 다른 전략이므로 기준선과 비교하지 말 것.")
    ap.add_argument("--out", default="reports", help="산출 디렉터리(기본 reports).")
    return ap.parse_args()


def _build_scheme(args: argparse.Namespace):
    """CLI 인자에서 창 분할 방식을 만든다."""
    common = dict(test_months=args.test_months, min_test_months=args.min_test_months)
    if args.scheme == "rolling":
        return RollingWindows(train_months=args.train_months, **common)
    return AnchoredWindows(min_train_months=args.min_train_months, **common)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    cfg, icfg = Config.load(), IRPConfig.load()
    start = args.start or icfg.start or cfg.start
    end = args.end or icfg.end or cfg.end
    switch_cost = cfg.cost if args.switch_cost is None else args.switch_cost

    # 후보 수십 개가 같은 시세·같은 지표를 반복해 읽는다 → 메모이제이션 래퍼로 주입(엔진 무수정).
    loader = MemoLoader(ParquetDataLoader(cfg.data_dir))
    indicator = MemoIndicator(TrendScoreIndicator())

    grid = CandidateGrid(axes=AXES_QUICK if args.grid == "quick" else AXES_FULL)
    scheme = _build_scheme(args)
    runner = CandidateRunner(loader=loader, indicator=indicator, cost=cfg.cost,
                             allow_missing=args.allow_missing)
    validator = WalkForwardValidator(runner=runner, grid=grid, scheme=scheme,
                                     rules=default_rules(), switch_cost=switch_cost)

    logger.info(f"구간 {start} ~ {end} · 격자 {args.grid} · 창 {args.scheme} · "
                f"검증 {args.test_months}개월 · 전환비용 {switch_cost * 100:.2f}%")
    results = validator.run(icfg, start=start, end=end)

    # 인샘플 기준: 동결 파라미터의 **전 구간** 성적(열화 판정의 분모).
    frozen_label = validator.frozen_label(runner.curves)
    is_metrics = curve_metrics(runner.curves[frozen_label]) if frozen_label else None

    verdicts: Dict[str, object] = {}
    for rule_name, res in results.items():
        logger.info("")
        for line in res.summary_lines():
            logger.info(line)
        v = judge(res, is_calmar=is_metrics["calmar"] if is_metrics else None)
        verdicts[rule_name] = v
        logger.info("")
        for line in v.lines():
            logger.info(line)

    logger.info("")
    for line in compare_rules(verdicts):
        logger.info(line)

    # 산출물 — 표는 규칙 전체, 그림은 대표 규칙(첫 번째 = plateau) 기준.
    rep = WalkForwardReport(args.out)
    head_name, head = next(iter(results.items()))
    rep.write_summary(results, is_metrics=is_metrics)
    rep.write_folds(head)
    rep.write_yearly(head)
    rep.write_verdicts(verdicts)
    rep.write_curves(head)
    rep.plot_curves(head)
    rep.plot_folds(head)
    rep.plot_spread(head, runner.curves)
    logger.info("")
    logger.info(f"산출물 기준 규칙 = {head_name} · 나머지 규칙은 요약·판정 CSV 에 포함")

    if is_metrics is not None:
        logger.info(f"인샘플 참고 · 동결 전 구간 CAGR {is_metrics['cagr_pct']:.1f}% · "
                    f"Calmar {is_metrics['calmar']:.2f} · MDD {is_metrics['mdd_pct']:.1f}% "
                    f"({LABEL_FROZEN})")


if __name__ == "__main__":
    main()
