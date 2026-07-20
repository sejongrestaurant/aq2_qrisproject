"""발표자료용 수치·차트 일괄 생성.

동결 브랜치(v2-tier2a-freeze)의 엔진으로 V1/V2/벤치마크를 재현하고,
발표자료와 제안서가 **같은 한 벌의 숫자**를 쓰도록 JSON 과 PNG 로 내보낸다.
구간은 제안서 표방 구간(config.end=2026-06-30)을 그대로 따른다.

산출물: _deck/metrics.json · _deck/*.png
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from data import ParquetDataLoader
from indicator import TrendScoreIndicator
from irp import IRPConfig
from irp.backtester_v2 import IRPBacktesterV2
from satellite.backtester_v2 import SatelliteBacktesterV2

# ── 표기 통일 ────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에서 마이너스 깨짐 방지

NAVY, GOLD, GREY = "#1F3A5F", "#C8A24B", "#9AA5B1"
OUT = Path("_deck")
OUT.mkdir(exist_ok=True)

FROZEN_RAMP = (52, 60, 0.3)


# ── 조립 ────────────────────────────────────────────────────────
def build(cost: float, ramp=FROZEN_RAMP) -> IRPBacktesterV2:
    """동결 파라미터로 백테스터를 만든다. ramp=None 이면 V1 원설계(이진 게이트)."""
    loader = ParquetDataLoader(Config.load().data_dir)
    ind = TrendScoreIndicator()
    if ramp is None:
        return IRPBacktesterV2(loader=loader, indicator=ind, cost=cost)
    lo, full, floor = ramp
    sat = SatelliteBacktesterV2(loader=loader, indicator=ind, cost=cost,
                                ramp_score=lo, full_score=full, entry_gate=None,
                                ramp_floor=floor, ramp_hold=True)
    return IRPBacktesterV2(loader=loader, indicator=ind, cost=cost, satellite=sat)


def run(cost: float = 0.0010, ramp=FROZEN_RAMP, end: str | None = None):
    cfg, icfg = Config.load(), IRPConfig.load()
    return build(cost, ramp).run(icfg, start=icfg.start, end=end or cfg.end)


def pack(res) -> dict:
    """지표 dict 를 발표자료가 쓸 최소 형태로 축약."""
    s, b = res.metrics["strategy"], res.metrics["benchmark"]
    keys = ("cagr_pct", "sharpe", "mdd_pct", "calmar")
    return {"strategy": {k: round(float(s[k]), 2) for k in keys},
            "benchmark": {k: round(float(b[k]), 2) for k in keys},
            "yearly": [{"year": r["year"],
                        "strat": round(r["strat_pct"], 1),
                        "bench": round(r["bench_pct"], 1),
                        "excess": round(r["excess_pct"], 1)} for r in res.yearly()]}


print("· 백테스트 재현 중...")
r_v2 = run()
r_v1 = run(ramp=None)
r_c3 = run(cost=0.0030)
r_cut = run(end="2025-12-31")
r_cut_v1 = run(ramp=None, end="2025-12-31")

M = {"window": {"start": IRPConfig.load().start, "end": Config.load().end},
     "cost_roundtrip": 0.0010,
     "v2": pack(r_v2), "v1": pack(r_v1),
     "cost_3x": pack(r_c3), "cut2025_v2": pack(r_cut), "cut2025_v1": pack(r_cut_v1)}

# ── 차트 1: 자산곡선 ────────────────────────────────────────────
eq, bh = r_v2.equity, r_v2.benchmark
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(eq.index, eq / eq.iloc[0], color=NAVY, lw=2.4, label="헬름 IRP세븐서티액티브", zorder=3)
ax.plot(bh.index, bh / bh.iloc[0], color=GREY, lw=1.8, label="KODEX TRF7030(벤치마크)", zorder=1)
ax.set_ylabel("누적 성장 (배)")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=.25)
ax.set_title("자산곡선 — 2020.01 ~ 2026.06 (거래비용 왕복 0.10% 반영)", fontsize=13, pad=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "equity.png", dpi=160); plt.close(fig)

# ── 차트 2: 낙폭(방어력의 핵심 근거) ─────────────────────────────
def dd(c: pd.Series) -> pd.Series:
    return (c / c.cummax() - 1) * 100

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.fill_between(bh.index, dd(bh), 0, color=GREY, alpha=.55, label=f"벤치마크 (MDD {dd(bh).min():.1f}%)")
ax.fill_between(eq.index, dd(eq), 0, color=NAVY, alpha=.75, label=f"본 상품 (MDD {dd(eq).min():.1f}%)")
ax.set_ylabel("고점 대비 낙폭 (%)")
ax.legend(loc="lower left", frameon=False)
ax.grid(alpha=.25)
ax.set_title("낙폭 비교 — 같은 시장에서 얼마나 덜 잃었나", fontsize=13, pad=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "drawdown.png", dpi=160); plt.close(fig)

# ── 차트 3: 연도별 ──────────────────────────────────────────────
ys = M["v2"]["yearly"]
x = np.arange(len(ys)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(x - w/2, [r["strat"] for r in ys], w, color=NAVY, label="본 상품")
ax.bar(x + w/2, [r["bench"] for r in ys], w, color=GREY, label="벤치마크")
ax.axhline(0, color="#333", lw=.9)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['year']}" if r["year"] != 2026 else "2026\n상반기" for r in ys])
ax.set_ylabel("연간 수익률 (%)")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.25)
ax.set_title("연도별 수익률 — 2022년 방어가 이 상품의 정체성", fontsize=13, pad=12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "yearly.png", dpi=160); plt.close(fig)

# ── 차트 4: 실효 노출률 (run_exposure.py 산출 CSV 재사용) ─────────
csv = Path("reports/exposure_monthly.csv")
if csv.exists():
    ex = pd.read_csv(csv)
    col = next((c for c in ex.columns if "exposure" in c.lower() or "노출" in c), None)
    dcol = next((c for c in ex.columns if "date" in c.lower() or "월" in c or "month" in c.lower()), ex.columns[0])
    if col:
        d = pd.to_datetime(ex[dcol])
        v = pd.to_numeric(ex[col], errors="coerce")
        v = v * 100 if v.max() <= 1.5 else v
        fig, ax = plt.subplots(figsize=(11, 4.0))
        ax.fill_between(d, v, 0, color=NAVY, alpha=.8)
        ax.axhline(70, color=GOLD, ls="--", lw=1.3, label="만충 = 70%")
        ax.axhline(float(v.mean()), color="#C0392B", ls=":", lw=1.5,
                   label=f"평균 {v.mean():.1f}%")
        ax.set_ylabel("위험자산 실효 노출 (%)")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=.25)
        ax.set_title("실효 노출률 — 위기에 스스로 내려간다 (2020.05 = 0%)", fontsize=13, pad=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(OUT / "exposure.png", dpi=160); plt.close(fig)
        M["exposure"] = {"mean_pct": round(float(v.mean()), 1),
                         "min_pct": round(float(v.min()), 1),
                         "checks": int(v.notna().sum())}

(OUT / "metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(M, ensure_ascii=False, indent=2))
print(f"\n· 차트 저장: {sorted(p.name for p in OUT.glob('*.png'))}")
