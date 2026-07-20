"""Run parameter experiments on top of the backtest engine."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtest.metrics import PerformanceConfig, calculate_performance_metrics
from src.experiments.analyzer import analyze_experiments
from src.experiments.config_generator import ExperimentConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentRunResult:
    """One experiment execution result."""

    experiment_id: str
    status: str
    config: dict[str, object]
    metrics: dict[str, object]
    error: str | None = None

    def to_row(self) -> dict[str, object]:
        """Return a flat result row for CSV/Parquet."""
        row = {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "error": self.error,
        }
        row.update(_flatten_config(self.config))
        row.update(self.metrics)
        return row


def run_experiments(
    configs: list[ExperimentConfig],
    *,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    factor_scores: pd.DataFrame,
    market_data: pd.DataFrame,
    output_dir: str | Path,
    parallel: bool = False,
    max_workers: int | None = None,
    rerun_failed_only: bool = False,
) -> pd.DataFrame:
    """Run experiment configs and save results with their settings.

    If `rerun_failed_only=True`, the runner reads existing results and executes only experiments
    whose last saved status is `failed`.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    selected_configs = _filter_failed(configs, path) if rerun_failed_only else configs

    if parallel and len(selected_configs) > 1:
        run_results = _run_parallel(
            selected_configs,
            prices=prices,
            universe=universe,
            factor_scores=factor_scores,
            market_data=market_data,
            output_dir=path,
            max_workers=max_workers,
        )
    else:
        run_results = [
            _run_one(config, prices, universe, factor_scores, market_data, path)
            for config in selected_configs
        ]

    new_results = pd.DataFrame([result.to_row() for result in run_results])
    combined = _merge_existing(path, new_results, rerun_failed_only=rerun_failed_only)
    analyzed = analyze_experiments(combined)
    save_experiment_results(analyzed, path)
    return analyzed


def save_experiment_results(results: pd.DataFrame, output_dir: str | Path) -> None:
    """Save experiment comparison results to CSV and Parquet."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    results.to_csv(path / "experiment_results.csv", index=False)
    results.to_parquet(path / "experiment_results.parquet", index=False)


def _run_parallel(
    configs: list[ExperimentConfig],
    *,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    factor_scores: pd.DataFrame,
    market_data: pd.DataFrame,
    output_dir: Path,
    max_workers: int | None,
) -> list[ExperimentRunResult]:
    results: list[ExperimentRunResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one, config, prices, universe, factor_scores, market_data, output_dir
            ): config
            for config in configs
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_one(
    config: ExperimentConfig,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    factor_scores: pd.DataFrame,
    market_data: pd.DataFrame,
    output_dir: Path,
) -> ExperimentRunResult:
    experiment_dir = output_dir / config.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        period_metrics = {}
        period_results: dict[str, BacktestResult] = {}
        for period_name, period in [
            ("train", config.train_period),
            ("validation", config.validation_period),
            ("test", config.test_period),
        ]:
            result = run_backtest(
                prices,
                universe,
                _factor_scores_for_setting(factor_scores, config.factor_setting),
                market_data,
                config=BacktestConfig(
                    start_date=period[0],
                    end_date=period[1],
                    strategy=config.weight_method,
                    transaction_cost=config.transaction_cost,
                    target_size=config.target_size,
                    rebalance_frequency=config.rebalance_frequency,
                    regime_equity_weights=config.regime_equity_weights,
                    factor_setting=config.factor_setting,
                    strategy_name=f"{config.experiment_id}_{period_name}",
                ),
                output_dir=experiment_dir / period_name,
            )
            period_results[period_name] = result
            returns = pd.Series(
                result.daily_results["daily_return"].to_numpy(),
                index=pd.to_datetime(result.daily_results["date"]),
            )
            metrics = calculate_performance_metrics(
                returns,
                turnover=pd.Series(
                    result.daily_results["turnover"].to_numpy(), index=returns.index
                ),
                transaction_cost=pd.Series(
                    result.daily_results["transaction_cost"].to_numpy(),
                    index=returns.index,
                ),
                config=PerformanceConfig(min_observations=1),
            )
            period_metrics.update(_prefix_metrics(period_name, metrics))

        final_metrics = _test_metric_aliases(period_metrics)
        final_metrics.update(period_metrics)
        final_metrics["test_warning"] = config.metadata["test_warning"]
        return ExperimentRunResult(
            experiment_id=config.experiment_id,
            status="completed",
            config=config.to_dict(),
            metrics=final_metrics,
        )
    except Exception as exc:
        LOGGER.exception("Experiment failed: %s", config.experiment_id)
        return ExperimentRunResult(
            experiment_id=config.experiment_id,
            status="failed",
            config=config.to_dict(),
            metrics={},
            error=str(exc),
        )


def _factor_scores_for_setting(factor_scores: pd.DataFrame, factor_setting: str) -> pd.DataFrame:
    """Return factor scores for a setting when precomputed variants are present."""
    if "factor_setting" not in factor_scores.columns:
        return factor_scores
    filtered = factor_scores.loc[factor_scores["factor_setting"] == factor_setting].copy()
    return filtered if not filtered.empty else factor_scores


def _prefix_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if _is_scalar(value)}


def _test_metric_aliases(period_metrics: dict[str, object]) -> dict[str, object]:
    aliases = {
        "cagr": "test_cagr",
        "sharpe_ratio": "test_sharpe_ratio",
        "maximum_drawdown": "test_maximum_drawdown",
        "calmar_ratio": "test_calmar_ratio",
        "annual_turnover": "test_annual_turnover",
        "total_transaction_cost": "test_total_transaction_cost",
    }
    return {key: period_metrics.get(source, 0.0) for key, source in aliases.items()}


def _flatten_config(config: dict[str, object]) -> dict[str, object]:
    row = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                row[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, tuple):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool | pd.Timestamp)


def _filter_failed(configs: list[ExperimentConfig], output_dir: Path) -> list[ExperimentConfig]:
    existing = _read_existing(output_dir)
    if existing.empty or "status" not in existing.columns:
        return configs
    failed_ids = set(existing.loc[existing["status"] == "failed", "experiment_id"].astype(str))
    return [config for config in configs if config.experiment_id in failed_ids]


def _merge_existing(
    output_dir: Path,
    new_results: pd.DataFrame,
    *,
    rerun_failed_only: bool,
) -> pd.DataFrame:
    existing = _read_existing(output_dir)
    if existing.empty or not rerun_failed_only:
        return new_results
    replaced = existing.loc[
        ~existing["experiment_id"].astype(str).isin(new_results["experiment_id"].astype(str))
    ]
    return pd.concat([replaced, new_results], ignore_index=True, sort=False)


def _read_existing(output_dir: Path) -> pd.DataFrame:
    parquet_path = output_dir / "experiment_results.parquet"
    csv_path = output_dir / "experiment_results.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


__all__ = ["ExperimentRunResult", "run_experiments", "save_experiment_results"]
