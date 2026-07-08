"""파이프라인 진입점 — 일봉 스윙 전략 백테스트 → 전략 비교 HTML 리포트.

흐름(데이터 → 지표 → 전략 → 백테스트 → 리포트):

    DataLoader → [TrendScoreSwing, SuperTrendSwing] → Backtester → HTMLReporter

각 단계는 독립 모듈의 클래스로, 인터페이스만 맞으면 개별 교체가 가능하다(확장성). 유니버스의 모든 종목에
대해 활성 전략들을 백테스트하고, 종목별로 전략을 헤드투헤드 비교하는 HTML 리포트를 생성한다.

실행: python main.py
"""
from __future__ import annotations

from typing import List

from backtest import Backtester
from config import Config
from data import ParquetDataLoader, YFinanceDataLoader
from indicator import SuperTrendIndicator, TrendScoreIndicator
from report import HTMLReporter
from strategy import (RegimeGatedTrendScoreStrategy, RegimeTrendRiderStrategy,
                      SMASlopeROCStrategy, Strategy, SuperTrendSwingStrategy,
                      TrendScoreSwingStrategy)


class Pipeline:
    """설정을 받아 전체 백테스트 파이프라인을 실행하는 오케스트레이터.

    Args (생성자):
        config: `Config` 인스턴스(주입). 없으면 config.json 로드.
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config.load()
        self.loader = self._build_loader()
        self.strategies = self._build_strategies()
        self.engine = Backtester(cost=self.cfg.cost)
        self.reporter = HTMLReporter(
            entry=self.cfg.entry, exit=self.cfg.exit, adx_gate=self.cfg.adx_gate)

    # ── 조립 ────────────────────────────────────────────────────────
    def _build_loader(self):
        """config.source 에 맞는 시세 로더를 생성한다(parquet | yfinance)."""
        if self.cfg.source == "yfinance":
            warmup = max(self.cfg.warmup_bars, self.cfg.trend_score.min_len)
            print(f"[data] yfinance 소스 · 워밍업 {warmup}봉 선확보 · "
                  f"구간 {self.cfg.start or '전체'}~{self.cfg.end or '최신'}")
            return YFinanceDataLoader(
                start=self.cfg.start, end=self.cfg.end, warmup_bars=warmup,
                cache_dir=self.cfg.data_dir if self.cfg.cache else None)
        return ParquetDataLoader(self.cfg.data_dir)

    def _build_strategies(self) -> List[Strategy]:
        """config 에서 활성화된 전략 인스턴스 목록을 조립한다(비교 대상).

        TrendScore 는 손절 없는 기본을 항상 넣고, stops_enabled 면 ATR 손절 변형을 추가로 넣어
        "손절 유무" 를 헤드투헤드 비교한다. SuperTrend 는 대조군으로 선택 추가한다.
        """
        strategies: List[Strategy] = []
        if self.cfg.use_trend_score:
            if self.cfg.use_trend_score_base:
                strategies.append(self._make_trend_score(with_stops=False))
            if self.cfg.stops_enabled and (self.cfg.stop_loss_atr or self.cfg.trailing_atr):
                strategies.append(self._make_trend_score(with_stops=True))
        if self.cfg.use_supertrend:
            st = self.cfg.supertrend
            strategies.append(SuperTrendSwingStrategy(
                indicator=SuperTrendIndicator(st.atr_period, st.multiplier)))
        if self.cfg.use_regime_ts:
            strategies.append(self._make_regime_ts())
        if self.cfg.use_sma_slope:
            # 국면 3분할(각도=정규화 기울기 %/일): 상승>+0.03 / 횡보 / 하락<-0.03.
            # 진입=상승국면 AND ROC 가속(≥0), 청산=하락국면 진입(횡보는 홀드).
            strategies.append(SMASlopeROCStrategy(
                sma_len=20, roc_len=10, roc_smooth=3, roc_slope_len=1, roc_th=0.0,
                slope_enter_th=0.03, slope_exit_th=-0.03))
        if self.cfg.use_trendrider:
            # 3조 regime-trendrider v4 (EMA20/60 국면 + ADX>10, 샹들리에·B1 선제청산)
            strategies.append(RegimeTrendRiderStrategy())
        return strategies

    def _make_regime_ts(self) -> RegimeGatedTrendScoreStrategy:
        """레짐게이트 TrendScore 전략을 조립한다(진입=TS+ADX 게이트, 청산=v5.3 lifeline).

        TrendScore 지표는 config 파라미터로 맞춰 다른 전략과 동일 점수를 쓰게 하고,
        ADX 게이트도 config 의 adx_gate 를 재사용해 "동일 진입 규칙 + 청산만 교체" 비교가 되게 한다.
        """
        ts = self.cfg.trend_score
        indicator = TrendScoreIndicator(
            min_len=ts.min_len, rsi_period=ts.rsi_period, adx_period=ts.adx_period,
            ewmac_weight=ts.ewmac_weight, tsmom_weight=ts.tsmom_weight,
            rsi_weight=ts.rsi_weight, adx_penalty_max=ts.adx_penalty_max,
            adx_full_strength=ts.adx_full_strength, smooth_span=ts.smooth_span)
        return RegimeGatedTrendScoreStrategy(
            adx_gate=self.cfg.adx_gate, adx_directional=self.cfg.adx_directional,
            adx_period=ts.adx_period, indicator=indicator)

    def _make_trend_score(self, with_stops: bool) -> TrendScoreSwingStrategy:
        """TrendScore 스윙 전략 1개를 조립한다(with_stops 면 ATR 손절 포함).

        지표 인스턴스는 변형마다 새로 만든다(상태 없음이라 무해하나 명확성 위해 분리).
        """
        ts = self.cfg.trend_score
        indicator = TrendScoreIndicator(
            min_len=ts.min_len, rsi_period=ts.rsi_period, adx_period=ts.adx_period,
            ewmac_weight=ts.ewmac_weight, tsmom_weight=ts.tsmom_weight,
            rsi_weight=ts.rsi_weight, adx_penalty_max=ts.adx_penalty_max,
            adx_full_strength=ts.adx_full_strength, smooth_span=ts.smooth_span)
        return TrendScoreSwingStrategy(
            entry=self.cfg.entry, exit=self.cfg.exit, adx_gate=self.cfg.adx_gate,
            adx_directional=self.cfg.adx_directional, indicator=indicator,
            atr_period=self.cfg.stop_atr_period,
            atr_stop_loss=(self.cfg.stop_loss_atr if with_stops else None),
            atr_trailing=(self.cfg.trailing_atr if with_stops else None))

    # ── 실행 ────────────────────────────────────────────────────────
    def run(self) -> str:
        """유니버스 × 활성전략을 백테스트하고 비교 HTML 리포트를 생성해 경로를 반환한다."""
        names = " vs ".join(s.name for s in self.strategies)
        print(f"[run] 전략: {names}\n")

        results = []
        for code in self.cfg.codes:
            try:
                price = self.loader.load(code)
                price.name = self.cfg.universe.get(code)  # config 표시명을 리포트에 반영
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {code}: 로드 실패 ({exc})")
                continue
            if len(price) < self.cfg.min_bars:
                print(f"[skip] {code}: 데이터 부족 ({len(price)} < {self.cfg.min_bars})")
                continue

            for strat in self.strategies:
                result = self.engine.run(price, strat, start=self.cfg.start, end=self.cfg.end)
                results.append(result)
                m = result.metrics["strategy"]
                print(f"[ok]   {code:<5} {strat.name:<26} 총수익 {m['total_return_pct']:>7.1f}%  "
                      f"CAGR {m['cagr_pct']:>5.1f}%  Sharpe {m['sharpe']:>4.2f}  "
                      f"MDD {m['mdd_pct']:>6.1f}%  거래 {m['n_trades']:>3}")

        if not results:
            raise RuntimeError("백테스트된 종목이 없습니다. data_dir/universe 를 확인하세요.")

        path = self.reporter.generate(
            results, self.cfg.out_path,
            title="일봉 스윙 전략 비교 — TrendScore vs SuperTrend")
        print(f"\n리포트 생성: {path}")
        self._print_universe_summary(results)
        return path

    def _print_universe_summary(self, results) -> None:
        """콘솔에 전략별 유니버스 평균(CAGR·Sharpe·MDD)을 출력한다."""
        import numpy as np
        by_strat: dict = {}
        for r in results:
            by_strat.setdefault(r.strategy_name, []).append(r.metrics["strategy"])
        bh = [r.metrics["benchmark"] for r in results]
        print("\n=== 유니버스 평균 ===")
        print(f"{'전략':<26}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}")
        for name, ms in by_strat.items():
            print(f"{name:<26}{np.mean([m['cagr_pct'] for m in ms]):>8.1f}"
                  f"{np.mean([m['sharpe'] for m in ms]):>8.2f}"
                  f"{np.mean([m['mdd_pct'] for m in ms]):>8.1f}")
        # B&H 는 종목당 전략수만큼 중복되므로 유니크 종목 기준 평균
        seen, uniq = set(), []
        for r in results:
            if r.code not in seen:
                seen.add(r.code); uniq.append(r.metrics["benchmark"])
        print(f"{'Buy&Hold':<26}{np.mean([m['cagr_pct'] for m in uniq]):>8.1f}"
              f"{np.mean([m['sharpe'] for m in uniq]):>8.2f}"
              f"{np.mean([m['mdd_pct'] for m in uniq]):>8.1f}")


def main() -> None:
    """config.json 을 로드해 파이프라인을 실행한다."""
    Pipeline(Config.load()).run()


if __name__ == "__main__":
    main()
