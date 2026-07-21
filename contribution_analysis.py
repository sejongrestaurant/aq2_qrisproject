"""동일가중 희석 가설 검증 — 종목별 구간 기여 분해(동결 V2 기준).

각 교체 구간에서 편입 종목 각각의 실제 수익률을 계산해:
  ① 같은 구간 동반 보유 종목 대비 상대 성과 (시기 교란 제거)
  ② 종목/자산군별 '동반 대비 평균 초과' 랭킹 → 상습 희석범 식별
  ③ 반사실: 특정 자산(예: 리츠)을 빼고 나머지 균등 재배분 시 슬리브 누적수익 변화

**기준은 동결 V2**(경사 52→60·바닥 30%)다 — 제안서가 파는 상품의 로테이션 기록으로 분해해야
전시물이 서로 어긋나지 않는다. 원본 V1 로 돌리면 교체 이력(rotations_log)이 이진 게이트의
것이라 상품과 다른 종목·시점을 분해하게 된다.

사용법: 저장소 루트에서
    uv run python contribution_analysis.py            # 전체 분해
    uv run python contribution_analysis.py 329200     # 해당 코드 제외 반사실 포함
출력: reports/contribution_by_pick.csv + 로그 요약
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from run_v2 import FROZEN_RAMP
from satellite.backtester_v2 import SatelliteBacktesterV2

logger = logging.getLogger("contribution_analysis")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    exclude = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = Config.load()
    icfg = IRPConfig.load()
    loader = ParquetDataLoader(cfg.data_dir)
    lo, full, floor = FROZEN_RAMP
    sat = SatelliteBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                                ramp_score=lo, full_score=full, ramp_floor=floor, ramp_hold=True)
    bt = IRPBacktesterV2(loader=loader, indicator=TrendScoreIndicator(), cost=cfg.cost,
                         satellite=sat)
    res = bt.run(icfg, start=icfg.start or cfg.start, end=icfg.end or cfg.end)
    log = res.rotations_log or []
    # 마지막 구간(t1=None)의 상한. 종목별 원본 마지막 봉까지 흘러가면(예: 411060 은 캐시가 하루
    # 더 길다) 백테스트 창 밖·종목마다 다른 날짜로 수익이 계산돼 분해가 오염된다. 엔진이 실제로
    # 쓴 마지막 거래일로 잘라, 모든 종목을 같은 창에서 본다.
    last_day = res.equity.index[-1]

    closes: dict[str, pd.Series] = {}

    def seg_ret(code: str, t0, t1) -> float | None:
        if code not in closes:
            try:
                closes[code] = loader.load(code).df["close"]
            except Exception as exc:  # noqa: BLE001 — 엔진과 같은 관용구(warn-and-skip)
                # 조용히 삼키면 분해가 일부 종목 빠진 채 '권위 있는' 수치로 나간다. 엔진
                # (satellite.backtester)과 동일하게 보이게 경고하고 건너뛴다.
                logger.warning(f"{code}: 종가 로드 실패 → 분해에서 제외 ({exc})")
                closes[code] = pd.Series(dtype=float)
        s = closes[code].loc[t0:t1]
        return None if len(s) < 2 else float(s.iloc[-1] / s.iloc[0] - 1)

    rows = []
    for i, r in enumerate(log):
        codes = [lb.split("·")[0] for lb in r["labels"]]
        if not codes:
            continue
        t0 = r["date"]
        t1 = log[i + 1]["date"] if i + 1 < len(log) else last_day
        rets = {c: seg_ret(c, t0, t1) for c in codes}
        rets = {c: v for c, v in rets.items() if v is not None}
        if not rets:
            continue
        mean_all = sum(rets.values()) / len(rets)
        for c, v in rets.items():
            others = [x for k, x in rets.items() if k != c]
            rows.append({
                "date": t0, "code": c,
                "label": next(lb for lb in r["labels"] if lb.startswith(c)),
                "ret": v,
                "vs_peers": v - (sum(others) / len(others)) if others else 0.0,
                "seg_mean": mean_all,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        # 로테이션이 없거나 전 구간 수익 계산 불가 — 빈 groupby 는 KeyError 로 죽으니 먼저 멈춘다.
        logger.error("분해할 구간이 없습니다(로테이션 0 또는 전 구간 종가 부족). 산출 중단.")
        return

    os.makedirs("reports", exist_ok=True)  # reports/ 는 gitignore — 새 클론에서 없을 수 있다
    df.to_csv("reports/contribution_by_pick.csv", index=False, encoding="utf-8-sig")
    logger.info(f"CSV 저장 · reports/contribution_by_pick.csv ({len(df)}행)")

    logger.info("=== 종목별 '동반 보유 대비' 평균 초과수익 (3회 이상 편입, 하위 10) ===")
    g = df.groupby("label").agg(n=("ret", "size"), avg_ret=("ret", "mean"),
                                avg_vs_peers=("vs_peers", "mean"))
    g = g[g["n"] >= 3].sort_values("avg_vs_peers")
    out = g.copy()
    out["avg_ret"] = (out["avg_ret"] * 100).round(2)
    out["avg_vs_peers"] = (out["avg_vs_peers"] * 100).round(2)
    for line in out.head(10).to_string().splitlines():
        logger.info(line)
    logger.info("=== 상위 10 (동반 대비 잘 벌어준 종목) ===")
    for line in out.tail(10).iloc[::-1].to_string().splitlines():
        logger.info(line)

    if exclude:
        # 반사실: exclude 코드를 빼고 나머지 균등 재배분한 구간수익으로 누적 비교
        base, cf = 1.0, 1.0
        for t0, seg in df.groupby("date"):
            base *= 1 + seg["ret"].mean()
            kept = seg[seg["code"] != exclude]
            cf *= 1 + (kept["ret"].mean() if len(kept) else seg["ret"].mean())
        logger.info(f"=== 반사실: {exclude} 제외 재배분 ===")
        logger.info(f"실제 슬리브 누적(구간 합성): {(base-1)*100:+.1f}%")
        logger.info(f"{exclude} 제외 시:            {(cf-1)*100:+.1f}%")
        logger.info(f"차이: {(cf-base)*100:+.1f}%p  (양수면 제외가 유리했다는 뜻)")
        logger.info("주: 거래비용·게이트 재계산 없는 근사치. 방향 판단용.")


if __name__ == "__main__":
    main()
