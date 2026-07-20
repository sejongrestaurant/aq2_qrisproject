"""Stage adapters for the end-to-end MUST30 pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.report import build_performance_report, export_performance_report
from src.config.settings import get_settings
from src.factors.composite_score import calculate_composite_scores
from src.pipeline.collect_fundamentals import collect_fundamentals
from src.pipeline.collect_prices import collect_prices
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.optimizer import WeightConstraints
from src.portfolio.rebalance import calculate_rebalance_weights
from src.portfolio.selector import select_monthly_portfolio
from src.regime.market_regime import classify_market_regime
from src.universe.loader import load_universe
from src.validation.backtest_audit import AuditConfig, run_backtest_audit

PIPELINE_STAGES: tuple[str, ...] = (
    "validate_universe",
    "collect_prices",
    "collect_fundamentals",
    "validate_data",
    "calculate_factors",
    "calculate_regime",
    "select_portfolio",
    "calculate_weights",
    "run_backtest",
    "generate_report",
)


@dataclass(frozen=True)
class PipelineContext:
    """Configuration and paths shared by all pipeline stages."""

    start_date: date
    end_date: date
    universe_path: Path = Path("data/universe/korea_active_etf_universe_100.csv")
    database_url: str | None = None
    output_dir: Path = Path("outputs/pipeline")
    input_dir: Path = Path("data")
    strategy: str = "score_weight"
    transaction_cost: float = 0.002
    target_size: int = 30
    full_refresh: bool = False
    collection_failure_threshold: float = 0.10

    def output_path(self, *parts: str) -> Path:
        """Return an output path under the configured pipeline output directory."""
        return self.output_dir.joinpath(*parts)

    def input_path(self, *parts: str) -> Path:
        """Return an input path under the configured data directory."""
        return self.input_dir.joinpath(*parts)


@dataclass(frozen=True)
class StageResult:
    """Normalized result returned by every pipeline stage."""

    stage_name: str
    message: str
    output_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, int | float | str | bool] = field(default_factory=dict)


StageFunction = Callable[[PipelineContext], StageResult]


def build_stage_registry() -> dict[str, StageFunction]:
    """Return the default stage registry keyed by stage name."""
    return {
        "validate_universe": validate_universe_stage,
        "collect_prices": collect_prices_stage,
        "collect_fundamentals": collect_fundamentals_stage,
        "validate_data": validate_data_stage,
        "calculate_factors": calculate_factors_stage,
        "calculate_regime": calculate_regime_stage,
        "select_portfolio": select_portfolio_stage,
        "calculate_weights": calculate_weights_stage,
        "run_backtest": run_backtest_stage,
        "generate_report": generate_report_stage,
    }


def validate_universe_stage(context: PipelineContext) -> StageResult:
    """Validate and summarize the configured point-in-time universe file."""
    universe = load_universe(context.universe_path)
    summary_path = context.output_path("universe_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    market_counts = universe["market"].value_counts().sort_index().to_dict()
    summary_path.write_text(
        pd.Series(
            {
                "row_count": int(len(universe)),
                "active_count": int(universe["is_active"].sum()),
                "market_counts": {str(key): int(value) for key, value in market_counts.items()},
            }
        ).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage_name="validate_universe",
        message="Universe validation completed.",
        output_paths=[summary_path],
        metrics={"row_count": int(len(universe)), "active_count": int(universe["is_active"].sum())},
    )


def collect_prices_stage(context: PipelineContext) -> StageResult:
    """Collect daily prices with incremental upsert semantics."""
    stats = collect_prices(
        universe_path=context.universe_path,
        start_date=context.start_date,
        end_date=context.end_date,
        database_url=context.database_url,
        full_refresh=context.full_refresh,
    )
    return StageResult(
        stage_name="collect_prices",
        message="Price collection completed.",
        metrics=_stats_to_metrics(stats),
    )


def collect_fundamentals_stage(context: PipelineContext) -> StageResult:
    """Collect point-in-time DART fundamentals for the pipeline date range."""
    stats = collect_fundamentals(
        universe_path=context.universe_path,
        start_year=context.start_date.year,
        end_year=context.end_date.year,
        database_url=context.database_url,
    )
    return StageResult(
        stage_name="collect_fundamentals",
        message="Fundamental collection completed.",
        metrics=_stats_to_metrics(stats),
    )


def validate_data_stage(context: PipelineContext) -> StageResult:
    """Run available bias and data-quality checks before downstream calculations."""
    prices = _read_optional_table(context.input_path("prices.parquet"))
    universe = load_universe(context.universe_path)
    factor_scores = _read_optional_table(context.input_path("factor_scores.parquet"))
    audit_dir = context.output_path("validation")
    report = run_backtest_audit(
        prices=prices,
        universe=universe,
        factor_scores=factor_scores,
        config=AuditConfig(mode="warning", halt_on_failure=False),
        output_dir=audit_dir,
    )
    failed_count = int((~report["passed"].astype(bool)).sum()) if not report.empty else 0
    return StageResult(
        stage_name="validate_data",
        message="Data validation completed.",
        output_paths=[audit_dir / "backtest_audit.csv", audit_dir / "backtest_audit.json"],
        metrics={"check_count": int(len(report)), "failed_count": failed_count},
    )


def calculate_factors_stage(context: PipelineContext) -> StageResult:
    """Calculate composite factor scores from prepared raw factor inputs."""
    raw_path = context.input_path("processed", "factor_inputs.parquet")
    if not raw_path.exists():
        existing = context.input_path("factor_scores.parquet")
        if existing.exists() and not context.full_refresh:
            return StageResult(
                stage_name="calculate_factors",
                message="Existing factor_scores.parquet reused.",
                output_paths=[existing],
                metrics={"reused_existing": True},
            )
        raise FileNotFoundError(f"Factor input file not found: {raw_path}")

    factor_inputs = pd.read_parquet(raw_path)
    factor_scores = calculate_composite_scores(factor_inputs)
    output_path = context.output_path("factor_scores.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factor_scores.to_parquet(output_path, index=False)
    return StageResult(
        stage_name="calculate_factors",
        message="Composite factor calculation completed.",
        output_paths=[output_path],
        metrics={"row_count": int(len(factor_scores))},
    )


def calculate_regime_stage(context: PipelineContext) -> StageResult:
    """Calculate market regime history from prepared market data."""
    market_data = _read_required_table(context.input_path("market_data.parquet"))
    prices = _read_optional_table(context.input_path("prices.parquet"))
    regime = classify_market_regime(market_data, prices)
    output_path = context.output_path("regime_history.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regime.to_parquet(output_path, index=False)
    return StageResult(
        stage_name="calculate_regime",
        message="Market regime calculation completed.",
        output_paths=[output_path],
        metrics={"row_count": int(len(regime))},
    )


def select_portfolio_stage(context: PipelineContext) -> StageResult:
    """Select the latest month-end MUST30 portfolio without using future rows."""
    universe = load_universe(context.universe_path)
    factor_scores = _read_factor_scores(context)
    prices = _read_optional_table(context.input_path("prices.parquet"))
    result = select_monthly_portfolio(
        factor_scores,
        universe,
        pd.Timestamp(context.end_date),
        constraints=PortfolioConstraints(target_size=context.target_size),
        price_history=prices,
    )
    selected_path = context.output_path("selected_portfolio.parquet")
    excluded_path = context.output_path("excluded_stocks.parquet")
    summary_path = context.output_path("constraint_summary.json")
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    result.selected_portfolio.to_parquet(selected_path, index=False)
    result.excluded_stocks.to_parquet(excluded_path, index=False)
    summary_path.write_text(
        pd.Series(result.constraint_summary).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage_name="select_portfolio",
        message="Portfolio selection completed.",
        output_paths=[selected_path, excluded_path, summary_path],
        metrics={"selected_count": int(len(result.selected_portfolio))},
    )


def calculate_weights_stage(context: PipelineContext) -> StageResult:
    """Calculate constrained stock and cash target weights for the selected portfolio."""
    selected = _read_required_table(context.output_path("selected_portfolio.parquet"))
    regime_history = _read_optional_table(context.output_path("regime_history.parquet"))
    regime = "Neutral"
    equity_weight = 0.8
    if regime_history is not None and not regime_history.empty:
        history = regime_history.copy()
        history["date"] = pd.to_datetime(history["date"])
        available = history.loc[history["date"] <= pd.Timestamp(context.end_date)]
        if not available.empty:
            row = available.sort_values("date").iloc[-1]
            regime = str(row["regime"])
            equity_weight = float(row.get("equity_weight", equity_weight))
    weights = calculate_rebalance_weights(
        selected,
        pd.Timestamp(context.end_date),
        regime=regime,
        method=context.strategy,  # type: ignore[arg-type]
        equity_weight=equity_weight,
        constraints=WeightConstraints(target_size=context.target_size),
    )
    output_path = context.output_path("target_weights.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(output_path, index=False)
    return StageResult(
        stage_name="calculate_weights",
        message="Target weight calculation completed.",
        output_paths=[output_path],
        metrics={"row_count": int(len(weights)), "equity_weight": float(equity_weight)},
    )


def run_backtest_stage(context: PipelineContext) -> StageResult:
    """Run the look-ahead-safe backtest engine and save output tables."""
    result = run_backtest(
        _read_required_table(context.input_path("prices.parquet")),
        load_universe(context.universe_path),
        _read_factor_scores(context),
        _read_required_table(context.input_path("market_data.parquet")),
        config=BacktestConfig(
            start_date=context.start_date.isoformat(),
            end_date=context.end_date.isoformat(),
            strategy=context.strategy,  # type: ignore[arg-type]
            transaction_cost=context.transaction_cost,
            target_size=context.target_size,
        ),
        output_dir=context.output_path("backtest"),
    )
    return StageResult(
        stage_name="run_backtest",
        message="Backtest completed.",
        output_paths=[context.output_path("backtest")],
        metrics={"daily_rows": int(len(result.daily_results))},
    )


def generate_report_stage(context: PipelineContext) -> StageResult:
    """Generate a performance report from daily backtest returns."""
    daily = _read_required_table(context.output_path("backtest", "daily_results.parquet"))
    if "daily_return" not in daily.columns:
        raise ValueError("daily_results.parquet must include daily_return")
    returns = pd.Series(
        pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0).to_numpy(),
        index=pd.to_datetime(daily["date"]),
        name="daily_return",
    )
    report = build_performance_report(returns)
    report_dir = context.output_path("report")
    export_performance_report(report, report_dir)
    return StageResult(
        stage_name="generate_report",
        message="Performance report generated.",
        output_paths=[report_dir],
        metrics={"return_observations": int(len(returns))},
    )


def _read_factor_scores(context: PipelineContext) -> pd.DataFrame:
    output_path = context.output_path("factor_scores.parquet")
    if output_path.exists():
        return pd.read_parquet(output_path)
    return _read_required_table(context.input_path("factor_scores.parquet"))


def _read_required_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _read_required_table(path)


def _stats_to_metrics(stats: object) -> dict[str, int | float | str | bool]:
    values: dict[str, int | float | str | bool] = {}
    for key, value in vars(stats).items():
        if isinstance(value, int | float | str | bool):
            values[key] = value
    return values


def default_context(
    *,
    start_date: date,
    end_date: date,
    universe_path: Path | None = None,
    database_url: str | None = None,
    output_dir: Path = Path("outputs/pipeline"),
    input_dir: Path = Path("data"),
    strategy: str = "score_weight",
    transaction_cost: float = 0.002,
    target_size: int = 30,
    full_refresh: bool = False,
    collection_failure_threshold: float = 0.10,
) -> PipelineContext:
    """Create a pipeline context using project settings as defaults."""
    settings = get_settings()
    return PipelineContext(
        start_date=start_date,
        end_date=end_date,
        universe_path=universe_path or settings.universe_csv_path,
        database_url=database_url or settings.database_url,
        output_dir=output_dir,
        input_dir=input_dir,
        strategy=strategy,
        transaction_cost=transaction_cost,
        target_size=target_size,
        full_refresh=full_refresh,
        collection_failure_threshold=collection_failure_threshold,
    )


__all__ = [
    "PIPELINE_STAGES",
    "PipelineContext",
    "StageFunction",
    "StageResult",
    "build_stage_registry",
    "default_context",
]
