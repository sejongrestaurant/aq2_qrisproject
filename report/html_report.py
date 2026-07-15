"""자체완결형 HTML 리포터.

백테스트 결과를 외부 의존 없는 단일 HTML 로 렌더링한다. 차트는 matplotlib 로 그려 PNG 를 base64 로
임베드하므로 파일 하나만 있으면 어디서든 열린다. 유니버스(복수 종목) 비교 표 + 종목별 상세 섹션
(성과 카드, 가격/보유구간, 자산곡선 vs Buy&Hold, TrendScore, 낙폭, 거래 테이블)을 생성한다.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경에서 렌더링
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from backtest import BacktestResult  # noqa: E402

from .base import Reporter  # noqa: E402

logger = logging.getLogger(__name__)

# 차트 제목·라벨에 쓰는 한글이 깨지지 않도록 한글 지원 폰트 후보(플랫폼별 우선순위)
_KOREAN_FONTS = ["Malgun Gothic", "NanumGothic", "AppleGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"]


def _configure_korean_font() -> None:
    """한글 지원 폰트를 matplotlib sans-serif 최우선으로 등록한다(임포트 시 1회).

    설치된 후보를 우선순위대로 찾아 sans-serif 목록 맨 앞에 두고, 기존 DejaVu Sans 등을 뒤에
    남긴다. 이렇게 하면 한글은 한글 폰트로, 라틴·기타 글자는 폴백 폰트로 글리프 단위 렌더된다.
    설치된 한글 폰트가 없으면 경고만 남기고 기본 폰트를 유지한다.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    korean = next((name for name in _KOREAN_FONTS if name in available), None)
    if korean is None:
        logger.warning("한글 지원 폰트를 찾지 못했습니다(%s) — 차트의 한글이 깨질 수 있습니다.",
                       ", ".join(_KOREAN_FONTS))
        return

    base = list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [korean] + [f for f in base if f != korean]
    # 일반 마이너스(축 눈금 등)는 ASCII 하이픈으로 그린다(한글 폰트엔 − U+2212 글리프가 없음).
    plt.rcParams["axes.unicode_minus"] = False
    # 로그축 지수(10⁻¹ 등)의 수학 마이너스는 mathtext 경로라 폴백이 안 먹어 한글 폰트에서 경고를
    # 쏟는다. 시각 영향이 거의 없는 벤치성 경고이므로 mathtext 로거만 조용히 시킨다.
    logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)


_configure_korean_font()

# 색상 팔레트(라이트 테마 기준, 접근성 고려한 대비)
_C_STRAT = "#2563eb"   # 전략 곡선(파랑)
_C_BENCH = "#94a3b8"   # 벤치마크(회색)
_C_HOLD = "#22c55e"    # 보유 구간 음영(초록)
_HOLD_ALPHA = 0.22     # 보유 구간 음영 진하기(0~1)
_C_PRICE = "#0f172a"   # 가격선
_C_SCORE = "#7c3aed"   # TrendScore(보라)
_C_ADX = "#0891b2"     # ADX(청록)
_C_DD = "#ef4444"      # 낙폭(빨강)
_C_ENTRY = "#16a34a"
_C_EXIT = "#dc2626"
_C_GATE = "#0e7490"    # ADX 게이트 가로선(진청록)


class HTMLReporter(Reporter):
    """백테스트 결과 → 자체완결 HTML 리포트.

    Args (생성자):
        entry / exit: 리포트에 표기하고 TrendScore 차트에 그릴 진입/청산 임계(전략과 일치시켜 전달).
                     None 이면 임계선 생략.
    """

    def __init__(self, entry: float | None = None, exit: float | None = None,
                 adx_gate: float | None = None):
        self.entry = entry
        self.exit = exit
        self.adx_gate = adx_gate

    # ── public ──────────────────────────────────────────────────────
    def generate(self, results: List[BacktestResult], out_path: str, *, title: str = "") -> str:
        """종목별 개별 HTML(전략 비교) + 링크 인덱스 페이지를 생성한다.

        같은 종목의 여러 전략 결과를 한 페이지(``<out_dir>/<code>.html``)에서 비교하고,
        ``out_path`` 인덱스에는 (종목×전략) 비교표와 각 종목 페이지 링크를 담는다.

        Args:
            results: 백테스트 결과 리스트(종목×전략 조합, 순서 무관).
            out_path: 인덱스 페이지 경로(종목 페이지는 같은 디렉터리에 생성).
            title: 리포트 제목.
        Returns:
            인덱스 페이지의 절대 경로.
        """
        if not results:
            raise ValueError("HTMLReporter.generate: results 가 비어 있음")

        title = title or "TrendScore Swing Strategy — Backtest Report"
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)

        groups = self._group_by_code(results)
        # 종목별 비교 페이지
        for code, group in groups.items():
            label = group[0].label  # 표시명(코드·이름). 파일명/링크는 code 유지
            self._write_page(
                out_path=os.path.join(out_dir, f"{code}.html"),
                title=f"{label} — {title}",
                header=self._header(group, f"{label} 전략 비교",
                                    strategy_names=[r.strategy_name for r in group]),
                body=self._symbol_page(code, group) + self._back_link(),
            )

        # 인덱스 페이지(전체 비교표 + 링크)
        self._write_page(
            out_path=out_path,
            title=title,
            header=self._header(results, title,
                                strategy_names=self._unique_strategies(results)),
            body=self._summary_table(groups),
        )
        return os.path.abspath(out_path)

    @staticmethod
    def _group_by_code(results: List[BacktestResult]) -> "OrderedDict[str, List[BacktestResult]]":
        """결과를 종목 코드별로 묶는다(첫 등장 순서 보존)."""
        from collections import OrderedDict
        groups: "OrderedDict[str, List[BacktestResult]]" = OrderedDict()
        for r in results:
            groups.setdefault(r.code, []).append(r)
        return groups

    @staticmethod
    def _unique_strategies(results: List[BacktestResult]) -> List[str]:
        seen = []
        for r in results:
            if r.strategy_name not in seen:
                seen.append(r.strategy_name)
        return seen

    # ── 종목 페이지(전략 비교) ───────────────────────────────────────
    def _symbol_page(self, code: str, group: List[BacktestResult]) -> str:
        """한 종목의 여러 전략을 비교하는 페이지 본문을 만든다."""
        compare = self._compare_block(code, group) if len(group) > 1 else ""
        details = "\n".join(
            self._result_section(r, heading=r.strategy_name) for r in group)
        return compare + details

    def _compare_block(self, code: str, group: List[BacktestResult]) -> str:
        """전략 비교 표 + 자산곡선 오버레이 차트."""
        chart = self._compare_equity_chart(group)
        table = self._compare_table(group)
        return f"""
        <section class="card">
          <h2>{_esc(code)} 전략 비교 <span class="sub">(전략 vs Buy&amp;Hold)</span></h2>
          {table}
          <img class="chart" src="data:image/png;base64,{chart}" alt="{_esc(code)} equity comparison"/>
        </section>"""

    def _compare_table(self, group: List[BacktestResult]) -> str:
        """전략별 핵심 지표 + Buy&Hold 한 줄."""
        rows = []
        for r in sorted(group, key=lambda x: x.metrics["strategy"]["sharpe"], reverse=True):
            s = r.metrics["strategy"]
            rows.append(
                "<tr>"
                f"<td class='code'>{_esc(r.strategy_name)}</td>"
                f"{_num(s['total_return_pct'], '%', color=True)}"
                f"{_num(s['cagr_pct'], '%', color=True)}"
                f"{_num(s['sharpe'], '', digits=2, color=True)}"
                f"{_num(s['mdd_pct'], '%', color=True)}"
                f"<td>{s.get('n_trades', 0)}</td>"
                f"{_num(s.get('win_pct', 0), '%')}"
                f"{_num(s.get('exposure_pct', 0), '%')}"
                "</tr>")
        b = group[0].metrics["benchmark"]  # B&H 는 종목 공통
        rows.append(
            "<tr class='bh'>"
            "<td class='code'>Buy&amp;Hold</td>"
            f"{_num(b['total_return_pct'], '%', color=True)}"
            f"{_num(b['cagr_pct'], '%', color=True)}"
            f"{_num(b['sharpe'], '', digits=2, color=True)}"
            f"{_num(b['mdd_pct'], '%', color=True)}"
            "<td>–</td><td>–</td>"
            f"{_num(100.0, '%')}"
            "</tr>")
        return f"""
        <div class="table-wrap">
        <table class="grid">
          <thead><tr><th>전략</th><th>총수익</th><th>CAGR</th><th>Sharpe</th><th>MDD</th>
            <th>거래</th><th>승률</th><th>노출</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        </div>"""

    def _compare_equity_chart(self, group: List[BacktestResult]) -> str:
        """전략들의 자산곡선 + Buy&Hold 오버레이(로그 스케일)."""
        fig, ax = plt.subplots(figsize=(11, 3.6))
        palette = [_C_STRAT, _C_SCORE, _C_ADX, "#f59e0b", "#db2777"]
        for i, r in enumerate(group):
            ax.plot(r.equity.index, r.equity, lw=1.5,
                    color=palette[i % len(palette)], label=r.strategy_name)
        b = group[0].benchmark
        ax.plot(b.index, b, lw=1.2, color=_C_BENCH, label="Buy&Hold")
        ax.set_yscale("log")
        ax.set_ylabel("Equity (×)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.15)
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        return self._fig_to_base64(fig)

    def generate_symbol(self, result: BacktestResult, out_path: str, *, title: str = "") -> str:
        """단일 종목·단일 전략 페이지만 생성한다(개별 호출용)."""
        title = title or "TrendScore Swing Strategy — Backtest Report"
        self._write_page(
            out_path=out_path,
            title=f"{result.label} — {title}",
            header=self._header([result], f"{result.label} · {result.strategy_name}"),
            body=self._result_section(result),
        )
        return os.path.abspath(out_path)

    def _write_page(self, out_path: str, title: str, header: str, body: str) -> None:
        """헤더+본문을 완성 HTML 로 감싸 파일에 쓴다."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        html = _PAGE.format(title=_esc(title), css=_CSS, header=header,
                            summary="", sections=body)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

    @staticmethod
    def _back_link() -> str:
        return '<p class="back"><a href="index.html">← 유니버스 인덱스로</a></p>'

    # ── 헤더 / 요약 ──────────────────────────────────────────────────
    def _header(self, results: List[BacktestResult], title: str,
                strategy_names: List[str] | None = None) -> str:
        r0 = results[0]
        span = f"{r0.price.index[0]:%Y-%m-%d} ~ {r0.price.index[-1]:%Y-%m-%d}"
        if self.entry is None:
            thr = "—"
        elif self.entry == self.exit:
            # 단일 라인 크로스(라이브 규칙): 돌파 진입 / 하회 청산
            thr = f"{self.entry:.0f} 돌파 진입 · {self.exit:.0f} 하회 청산"
        else:
            thr = f"진입 ≥ {self.entry:.0f} · 청산 &lt; {self.exit:.0f}"
        if self.adx_gate is not None:
            thr += f" · ADX ≥ {self.adx_gate:.0f}"

        names = strategy_names or [r0.strategy_name]
        strat_val = _esc(", ".join(names)) if len(names) <= 3 else f"{len(names)}개 전략"
        codes = self._unique_codes(results)
        chips = [
            ("전략", strat_val),
            ("기간", span),
            ("TrendScore 임계", thr),
            ("왕복비용", f"{r0.cost * 100:.2f}%"),
            (("종목", _esc(r0.label)) if len(codes) == 1 else ("종목 수", str(len(codes)))),
        ]
        chip_html = "".join(
            f'<div class="chip"><span class="k">{k}</span><span class="v">{v}</span></div>'
            for k, v in chips)
        return f"<h1>{_esc(title)}</h1><div class='chips'>{chip_html}</div>"

    @staticmethod
    def _unique_codes(results: List[BacktestResult]) -> List[str]:
        seen = []
        for r in results:
            if r.code not in seen:
                seen.append(r.code)
        return seen

    def _summary_table(self, groups) -> str:
        """인덱스 비교표: (종목 × 전략) 행 + 종목 공통 Buy&Hold. 종목명은 상세 페이지로 링크."""
        rows = []
        for code, group in groups.items():
            b = group[0].metrics["benchmark"]
            ordered = sorted(group, key=lambda x: x.metrics["strategy"]["sharpe"], reverse=True)
            for i, r in enumerate(ordered):
                s = r.metrics["strategy"]
                code_cell = (f"<a href='{_esc(code)}.html'>{_esc(r.label)}</a>" if i == 0 else "")
                cls = " class='grp'" if i == 0 else ""
                rows.append(
                    f"<tr{cls}>"
                    f"<td class='code'>{code_cell}</td>"
                    f"<td>{_esc(r.strategy_name)}</td>"
                    f"{_num(s['total_return_pct'], '%', color=True)}"
                    f"{_num(s['cagr_pct'], '%', color=True)}"
                    f"{_num(s['sharpe'], '', digits=2, color=True)}"
                    f"{_num(s['mdd_pct'], '%', color=True)}"
                    f"<td>{s.get('n_trades', 0)}</td>"
                    f"{_num(s.get('exposure_pct', 0), '%')}"
                    f"{_num(b['cagr_pct'], '%', color=True)}"
                    f"{_num(b['mdd_pct'], '%', color=True)}"
                    "</tr>")
        # 전략별 유니버스 평균(공정 비교)
        foot = self._strategy_averages(groups)
        return f"""
        <section class="card">
          <h2>유니버스 비교 <span class="sub">(종목 클릭 → 상세 · 종목 내 Sharpe 내림차순)</span></h2>
          <div class="table-wrap">
          <table class="grid">
            <thead><tr>
              <th>종목</th><th>전략</th><th>총수익</th><th>CAGR</th><th>Sharpe</th><th>MDD</th>
              <th>거래</th><th>노출</th><th>B&amp;H CAGR</th><th>B&amp;H MDD</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
            <tfoot>{foot}</tfoot>
          </table>
          </div>
        </section>"""

    def _strategy_averages(self, groups) -> str:
        """전략별 유니버스 평균(CAGR·Sharpe·MDD) 푸터 행들."""
        by_strategy: dict = {}
        for group in groups.values():
            for r in group:
                by_strategy.setdefault(r.strategy_name, []).append(r.metrics["strategy"])
        out = []
        for name, ms in by_strategy.items():
            cagr = np.mean([m["cagr_pct"] for m in ms])
            sharpe = np.mean([m["sharpe"] for m in ms])
            mdd = np.mean([m["mdd_pct"] for m in ms])
            out.append(
                f"<tr><td>평균</td><td>{_esc(name)}</td><td></td>"
                f"{_num(cagr, '%', color=True)}{_num(sharpe, '', digits=2, color=True)}"
                f"{_num(mdd, '%', color=True)}<td colspan='4'></td></tr>")
        return "".join(out)

    # ── 종목별 상세 섹션 ─────────────────────────────────────────────
    def _result_section(self, r: BacktestResult, heading: str | None = None) -> str:
        s = r.metrics["strategy"]
        b = r.metrics["benchmark"]
        cards = "".join([
            _stat("총수익", s["total_return_pct"], "%", color=True),
            _stat("CAGR", s["cagr_pct"], "%", color=True),
            _stat("Sharpe", s["sharpe"], "", digits=2, color=True),
            _stat("Sortino", s.get("sortino", 0), "", digits=2, color=True),
            _stat("Calmar", s.get("calmar", 0), "", digits=2, color=True),
            _stat("MDD", s["mdd_pct"], "%", color=True),
            _stat("Ulcer", s.get("ulcer", 0), "", digits=2),
            _stat("거래수", s.get("n_trades", 0), "", digits=0),
            _stat("승률", s.get("win_pct", 0), "%"),
            _stat("평균손익", s.get("avg_trade_pct", 0), "%", color=True),
            _stat("노출", s.get("exposure_pct", 0), "%"),
        ])
        vs = (f"vs {_esc(r.benchmark_name)}: CAGR {b['cagr_pct']:+.1f}% · MDD {b['mdd_pct']:.1f}% · "
              f"Sharpe {b['sharpe']:.2f} · Sortino {b.get('sortino', 0):.2f} · "
              f"Calmar {b.get('calmar', 0):.2f} · Ulcer {b.get('ulcer', 0):.2f}")
        heading = heading or r.code
        chart = self._chart(r)
        recovery = self._recovery_table(r)
        yearly = self._yearly_block(r)
        rotations = self._rotations_block(r)
        trades = self._trades_table(r)
        return f"""
        <section class="card">
          <div class="sec-head">
            <h2>{_esc(heading)}</h2>
            <span class="vs">{vs}</span>
          </div>
          <h3 class="block-title">전체 기간 성과</h3>
          <div class="stats">{cards}</div>
          {recovery}
          <img class="chart" src="data:image/png;base64,{chart}" alt="{_esc(r.code)} charts"/>
          {yearly}
          {rotations}
          {trades}
        </section>"""

    # ── 회복(언더워터) 분석 표 ───────────────────────────────────────
    def _recovery_table(self, r: BacktestResult) -> str:
        """전략 vs 벤치마크 회복 지표(평균·최장 회복일수, 연평균 신규 고점 갱신)."""
        s = r.metrics["strategy"]
        b = r.metrics["benchmark"]

        def row(label: str, key: str, unit: str, digits: int = 0) -> str:
            return (f"<tr><td class='code'>{label}</td>"
                    f"<td>{_fmt(s.get(key, 0), unit, digits)}</td>"
                    f"<td>{_fmt(b.get(key, 0), unit, digits)}</td></tr>")

        rows = (row("평균 회복일수", "avg_recovery_days", "일")
                + row("최장 회복일수", "max_recovery_days", "일")
                + row("연평균 신규 고점 갱신", "new_highs_per_year", "회", 1))
        return f"""
        <h3 class="block-title">회복 분석 (Recovery)</h3>
        <div class="table-wrap">
        <table class="grid small">
          <thead><tr><th>지표</th><th>전략</th><th>{_esc(r.benchmark_name)}</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        </div>"""

    # ── 로테이션(섹터 선정) 내역 ─────────────────────────────────────
    def _rotations_block(self, r: BacktestResult) -> str:
        """모멘텀 로테이션 선정 이력 표(그때그때 어떤 종목을 골랐는지).

        각 교체 시점의 날짜·종목 수·구간수익·선정 종목 목록을 보여준다. rotations_log 가 없는
        결과(단일종목 전략 등)에서는 아무것도 그리지 않는다.
        """
        log = r.rotations_log
        if not log:
            return ""
        rows = []
        prev: set = set()   # 직전 회차 보유 종목(신규 편입 강조용)
        for i, ev in enumerate(log, 1):
            d = ev["date"]
            date_txt = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else _esc(d)
            labels = ev.get("labels", [])
            # 직전 회차에 없던 종목만 붉은 볼드로 강조(그때그때 '새로 교체된' 종목 식별).
            picks = ", ".join(
                (f"<span class='new'>{_esc(x)}</span>" if x not in prev else _esc(x))
                for x in labels)
            prev = set(labels)
            rows.append(
                "<tr>"
                f"<td>{i}</td><td class='code'>{date_txt}</td>"
                f"<td>{ev.get('n', len(labels))}</td>"
                f"{_num(ev.get('ret_pct', 0.0), '%', color=True)}"
                f"<td class='picks'>{picks}</td>"
                "</tr>")
        return f"""
        <details class="trades" open>
          <summary>섹터 로테이션 내역 ({len(log)}회 · 교체 시점별 선정 종목 · <span class="new">붉은색</span>=신규 편입)</summary>
          <div class="table-wrap">
          <table class="grid small">
            <thead><tr><th>#</th><th>편입일</th><th>종목수</th><th>구간수익</th>
              <th>선정 종목</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
          </div>
        </details>"""

    # ── 연도별 성과 (표 + 막대차트) ─────────────────────────────────
    def _yearly_block(self, r: BacktestResult) -> str:
        rows_data = r.yearly()
        if not rows_data:
            return ""
        rows = []
        for y in rows_data:
            rows.append(
                "<tr>"
                f"<td class='code'>{y['year']}</td>"
                f"{_num(y['strat_pct'], '%', color=True)}"
                f"{_num(y['bench_pct'], '%', color=True)}"
                f"{_num(y['excess_pct'], '%', color=True)}"
                f"{_num(y['mdd_pct'], '%', color=True)}"
                f"<td>{y['n_trades']}</td>"
                f"{_num(y['win_pct'], '%')}"
                f"{_num(y['exposure_pct'], '%')}"
                "</tr>")
        chart = self._yearly_chart(rows_data)
        return f"""
        <h3 class="block-title">연도별 성과</h3>
        <img class="chart yearly" src="data:image/png;base64,{chart}" alt="yearly returns"/>
        <div class="table-wrap">
        <table class="grid small">
          <thead><tr><th>연도</th><th>전략</th><th>B&amp;H</th><th>초과</th>
            <th>연중MDD</th><th>거래</th><th>승률</th><th>노출</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        </div>"""

    def _yearly_chart(self, rows_data) -> str:
        """연도별 전략 vs Buy&Hold 수익률 그룹 막대차트."""
        years = [d["year"] for d in rows_data]
        strat = [d["strat_pct"] for d in rows_data]
        bench = [d["bench_pct"] for d in rows_data]
        x = np.arange(len(years))
        w = 0.4

        fig, ax = plt.subplots(figsize=(11, 3.0))
        ax.bar(x - w / 2, strat, w, label="Strategy", color=_C_STRAT)
        ax.bar(x + w / 2, bench, w, label="Buy&Hold", color=_C_BENCH)
        ax.axhline(0, color="#334155", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years, fontsize=8)
        ax.set_ylabel("Annual Return %")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, axis="y", alpha=0.15)
        ax.margins(x=0.01)
        return self._fig_to_base64(fig)

    def _trades_table(self, r: BacktestResult) -> str:
        if not r.trades:
            return "<p class='muted'>거래 없음</p>"
        rows = []
        for i, t in enumerate(r.trades, 1):
            d = t.as_dict()
            cls = "pos" if t.ret > 0 else "neg"
            rows.append(
                f"<tr><td>{i}</td><td>{d['entry_date']}</td><td>{d['exit_date']}</td>"
                f"<td>{d['bars_held']}</td><td>{d['entry_px']:.2f}</td><td>{d['exit_px']:.2f}</td>"
                f"<td class='{cls}'>{d['ret_pct']:+.2f}%</td><td>{d['exit_reason']}</td></tr>")
        return f"""
        <details class="trades">
          <summary>거래 내역 ({len(r.trades)}건)</summary>
          <div class="table-wrap">
          <table class="grid small">
            <thead><tr><th>#</th><th>진입일</th><th>청산일</th><th>보유봉</th>
              <th>진입가</th><th>청산가</th><th>손익</th><th>사유</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
          </div>
        </details>"""

    # ── 차트 (matplotlib → base64 PNG) ──────────────────────────────
    def _chart(self, r: BacktestResult) -> str:
        """적응형 차트: 가격(+오버레이·마커) · 자산곡선 · [오실레이터] · 낙폭.

        오실레이터형 지표(TrendScore/ADX)가 있으면 별도 패널을 넣고, 없으면(예: SuperTrend)
        해당 패널을 생략한다. 가격 수준 지표(SuperTrend 라인)는 가격 패널에 추세색으로 겹쳐 그린다.
        """
        df = r.price
        idx = df.index
        score, score_name, adx, adx_name = self._split_indicators(r.indicators)
        has_osc = score is not None or adx is not None

        # 패널 구성(오실레이터 유무에 따라 3단/4단)
        ratios = [2.4, 2.2] + ([1.6] if has_osc else []) + [1.2]
        fig, axes = plt.subplots(
            len(ratios), 1, figsize=(11, 3.1 + 2.3 * len(ratios)), sharex=True,
            gridspec_kw={"height_ratios": ratios, "hspace": 0.12})
        ax_px, ax_eq = axes[0], axes[1]
        ax_sc = axes[2] if has_osc else None
        ax_dd = axes[-1]

        # (1) 가격 + 보유 음영 + 오버레이(SuperTrend) + 매수/매도 마커
        ax_px.plot(idx, df["close"], color=_C_PRICE, lw=1.0, zorder=1)
        self._shade_holdings(ax_px, r.target_long)
        self._draw_overlays(ax_px, r)
        self._mark_trades(ax_px, r)
        ax_px.set_ylabel("Price")
        ax_px.set_title(f"{r.code} — {r.strategy_name}", loc="left", fontsize=11, fontweight="bold")
        ax_px.set_yscale("log")

        # (2) 자산곡선 vs Buy&Hold
        ax_eq.plot(idx, r.equity, color=_C_STRAT, lw=1.4, label="Strategy")
        ax_eq.plot(r.benchmark.index, r.benchmark, color=_C_BENCH, lw=1.2, label=r.benchmark_name)
        ax_eq.set_ylabel("Equity (×)")
        ax_eq.set_yscale("log")
        ax_eq.legend(loc="upper left", fontsize=8, frameon=False)
        self._shade_holdings(ax_eq, r.target_long)

        # (3) 오실레이터: TrendScore(좌축) + ADX(우축, 보조) + 기준 가로선
        if ax_sc is not None:
            self._draw_oscillator(ax_sc, idx, score, score_name, adx, adx_name, r.target_long)

        # (4) 낙폭(전략)
        dd = (r.equity / r.equity.cummax() - 1.0) * 100
        ax_dd.fill_between(idx, dd, 0, color=_C_DD, alpha=0.35)
        ax_dd.plot(idx, dd, color=_C_DD, lw=0.8)
        ax_dd.set_ylabel("Drawdown %")

        for ax in axes:
            ax.grid(True, alpha=0.15)
            ax.margins(x=0)
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        return self._fig_to_base64(fig)

    def _draw_oscillator(self, ax_sc, idx, score, score_name, adx, adx_name, target_long) -> None:
        """오실레이터 패널: TrendScore(좌축, 기준선) + ADX(우축 보조, 게이트선)."""
        self._shade_holdings(ax_sc, target_long)
        handles = []
        if score is not None:
            h_ts, = ax_sc.plot(idx, score, color=_C_SCORE, lw=1.1, label=score_name, zorder=3)
            handles.append(h_ts)
            if self.entry is not None:
                ax_sc.axhline(self.entry, color=_C_ENTRY, ls="--", lw=0.9, alpha=0.85)
                self._label_line(ax_sc, idx, self.entry, f"TS {self.entry:.0f}", _C_ENTRY)
            if self.exit is not None and self.exit != self.entry:
                ax_sc.axhline(self.exit, color=_C_EXIT, ls="--", lw=0.9, alpha=0.85)
                self._label_line(ax_sc, idx, self.exit, f"TS {self.exit:.0f}", _C_EXIT)
            ax_sc.set_ylim(0, 100)
            ax_sc.set_ylabel(score_name, color=_C_SCORE)
            ax_sc.tick_params(axis="y", labelcolor=_C_SCORE)
        if adx is not None:
            ax_ad = ax_sc.twinx()
            h_ad, = ax_ad.plot(idx, adx, color=_C_ADX, lw=0.9, alpha=0.85, label=adx_name, zorder=2)
            handles.append(h_ad)
            ax_ad.axhline(0, color="#94a3b8", lw=0.6, alpha=0.6)
            if self.adx_gate is not None:
                ax_ad.axhline(self.adx_gate, color=_C_GATE, ls=":", lw=1.0, alpha=0.9)
                self._label_line(ax_ad, idx, self.adx_gate, f"ADX≥{self.adx_gate:.0f}",
                                 _C_GATE, right=True)
            amax = float(np.nanmax(np.abs(adx.to_numpy()))) if len(adx) else 30.0
            ax_ad.set_ylim(-amax * 1.1, amax * 1.1)
            ax_ad.set_ylabel(adx_name, color=_C_ADX)
            ax_ad.tick_params(axis="y", labelcolor=_C_ADX)
            ax_ad.margins(x=0)
            ax_ad.grid(False)
        if handles:
            ax_sc.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    @staticmethod
    def _draw_overlays(ax_px, r: BacktestResult) -> None:
        """가격 수준 오버레이를 가격 패널에 겹쳐 그린다.

        · 'stop' 계열(ATR 손절선): 주황 점선 단일 라인(보유 중 손절 위치).
        · 그 외(SuperTrend 라인): 추세색(상승 초록/하락 빨강)으로 분리해 방향 전환 시각화.
        """
        if not r.overlays:
            return
        up = r.target_long.reindex(r.price.index).fillna(False).to_numpy()
        for name, series in r.overlays.items():
            s = series.reindex(r.price.index)
            if "stop" in name.lower():
                ax_px.plot(s.index, s, color="#f59e0b", lw=1.0, ls="--", alpha=0.9,
                           zorder=4, label=name)
            else:
                ax_px.plot(s.index, s.where(up), color=_C_ENTRY, lw=1.0, alpha=0.9, zorder=2)
                ax_px.plot(s.index, s.where(~up), color=_C_EXIT, lw=1.0, alpha=0.9, zorder=2,
                           label=name)

    @staticmethod
    def _split_indicators(indicators: dict):
        """지표 딕셔너리에서 TrendScore 계열과 ADX 계열을 분리한다.

        Returns: (score_series, score_name, adx_series|None, adx_name|None).
        키 이름에 'ADX' 가 포함되면 ADX, 그 외 첫 항목을 TrendScore 로 본다.
        """
        score = adx = None
        score_name, adx_name = "Indicator", None
        for name, series in indicators.items():
            if "ADX" in name.upper():
                adx, adx_name = series, name
            elif score is None:
                score, score_name = series, name
        return score, score_name, adx, adx_name

    @staticmethod
    def _label_line(ax, idx, y: float, text: str, color: str, right: bool = False) -> None:
        """가로 기준선에 값 라벨을 붙인다(좌측 기본, right=True 면 우측 정렬)."""
        x = idx[-1] if right else idx[0]
        ax.annotate(text, xy=(x, y), fontsize=7, color=color,
                    va="bottom", ha=("right" if right else "left"),
                    xytext=(-2 if right else 2, 1), textcoords="offset points")

    @staticmethod
    def _mark_trades(ax, r: BacktestResult) -> None:
        """가격 축에 매수(▲)·매도(▼) 시점을 실제 체결가(익일 시가) 위치로 표시한다."""
        if not r.trades:
            return
        buy_x = [t.entry_date for t in r.trades]
        buy_y = [t.entry_px for t in r.trades]
        sell_x = [t.exit_date for t in r.trades]
        sell_y = [t.exit_px for t in r.trades]
        ax.scatter(buy_x, buy_y, marker="^", s=46, color=_C_ENTRY, edgecolors="white",
                   linewidths=0.5, zorder=5, label="Buy")
        ax.scatter(sell_x, sell_y, marker="v", s=46, color=_C_EXIT, edgecolors="white",
                   linewidths=0.5, zorder=5, label="Sell")
        ax.legend(loc="upper left", fontsize=8, frameon=False)

    @staticmethod
    def _shade_holdings(ax, target_long) -> None:
        """target_long==True(보유 예정) 구간을 초록으로 얕게 음영한다."""
        vals = target_long.to_numpy()
        idx = target_long.index
        start = None
        for i, on in enumerate(vals):
            if on and start is None:
                start = idx[i]
            elif not on and start is not None:
                ax.axvspan(start, idx[i], color=_C_HOLD, alpha=_HOLD_ALPHA, lw=0)
                start = None
        if start is not None:
            ax.axvspan(start, idx[-1], color=_C_HOLD, alpha=_HOLD_ALPHA, lw=0)

    @staticmethod
    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ──────────────────────────────────────────────────────────────────────
# HTML/CSS 템플릿 + 포매팅 헬퍼
# ──────────────────────────────────────────────────────────────────────
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt(v, unit="", digits=1) -> str:
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return "—"
    return f"{v:,.{digits}f}{unit}"


def _num(v, unit="", digits=1, color=False) -> str:
    """비교 표 셀. color=True 면 부호에 따라 색을 입힌다."""
    txt = _fmt(v, unit, digits)
    if color and v is not None and v == v:
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
        return f"<td class='{cls}'>{txt}</td>"
    return f"<td>{txt}</td>"


def _stat(label, v, unit="", digits=1, color=False) -> str:
    """성과 카드 1개."""
    txt = _fmt(v, unit, digits)
    cls = ""
    if color and v is not None and v == v:
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    return (f'<div class="stat"><div class="stat-v {cls}">{txt}</div>'
            f'<div class="stat-k">{label}</div></div>')


_CSS = """
:root{--bg:#f8fafc;--card:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;
      --pos:#16a34a;--neg:#dc2626;--accent:#2563eb;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Malgun Gothic',sans-serif;
     line-height:1.5;padding:32px 20px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:24px;margin:0 0 14px}
h2{font-size:18px;margin:0 0 14px}
h2 .sub{font-size:12px;color:var(--muted);font-weight:400}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:6px 12px;font-size:13px;display:flex;gap:8px;align-items:center}
.chip .k{color:var(--muted)} .chip .v{font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:22px;margin-bottom:22px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;
          flex-wrap:wrap;gap:8px;margin-bottom:16px}
.sec-head h2{margin:0}
.vs{font-size:12px;color:var(--muted)}
h3.block-title{font-size:14px;margin:20px 0 12px;padding-bottom:6px;
               border-bottom:1px solid var(--line);color:var(--muted);
               text-transform:none;letter-spacing:.02em}
h3.block-title:first-of-type{margin-top:4px}
img.chart.yearly{margin-bottom:12px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat-v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.stat-k{font-size:12px;color:var(--muted);margin-top:2px}
.chart{width:100%;height:auto;border-radius:8px;margin-top:6px}
.table-wrap{overflow-x:auto}
table.grid{width:100%;border-collapse:collapse;font-size:13px;
           font-variant-numeric:tabular-nums}
table.grid.small{font-size:12px}
table.grid th{text-align:right;color:var(--muted);font-weight:600;
              padding:8px 10px;border-bottom:2px solid var(--line);white-space:nowrap}
table.grid td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}
table.grid th:first-child,table.grid td:first-child{text-align:left}
table.grid th:last-child,table.grid td.picks{text-align:left}
table.grid td.picks{color:var(--muted);font-size:12px}
.new{color:var(--neg);font-weight:700}
table.grid td.code{font-weight:600}
table.grid tfoot td{font-weight:700;border-top:2px solid var(--line);border-bottom:none}
table.grid tr.grp td{border-top:2px solid var(--line)}
table.grid tr.bh td{font-style:italic;color:var(--muted)}
.pos{color:var(--pos)} .neg{color:var(--neg)}
.muted{color:var(--muted);font-size:13px}
table.grid td.code a{color:var(--accent);text-decoration:none;font-weight:600}
table.grid td.code a:hover{text-decoration:underline}
p.back{margin:18px 0 0}
p.back a{color:var(--accent);text-decoration:none;font-size:13px}
p.back a:hover{text-decoration:underline}
details.trades{margin-top:8px}
details.trades summary{cursor:pointer;font-size:13px;color:var(--accent);
                       padding:8px 0;user-select:none}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:8px}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
"""

_PAGE = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{css}</style>
</head><body><div class="wrap">
{header}
{summary}
{sections}
<footer>Generated by EST Indicator · TrendScore Swing Backtest</footer>
</div></body></html>
"""
