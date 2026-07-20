"""CLI for running parameter experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiments.config_generator import generate_experiment_configs
from src.experiments.runner import run_experiments
from src.universe.loader import load_universe


def main(argv: list[str] | None = None) -> None:
    """Run the experiment grid from the command line."""
    parser = argparse.ArgumentParser(description="Run MUST30 parameter experiments")
    parser.add_argument("--prices", type=Path, default=Path("data/prices.parquet"))
    parser.add_argument(
        "--universe", type=Path, default=Path("data/universe/korea_active_etf_universe_100.csv")
    )
    parser.add_argument("--factor-scores", type=Path, default=Path("data/factor_scores.parquet"))
    parser.add_argument("--market-data", type=Path, default=Path("data/market_data.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--rerun-failed-only", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional smoke-test limit for configs"
    )
    args = parser.parse_args(argv)

    configs = generate_experiment_configs()
    if args.limit is not None:
        configs = configs[: args.limit]

    results = run_experiments(
        configs,
        prices=_read_table(args.prices),
        universe=load_universe(args.universe)
        if args.universe.suffix.lower() == ".csv"
        else _read_table(args.universe),
        factor_scores=_read_table(args.factor_scores),
        market_data=_read_table(args.market_data),
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_workers=args.max_workers,
        rerun_failed_only=args.rerun_failed_only,
    )
    top = results.head(10).loc[
        :, ["experiment_id", "status", "multi_criteria_score", "consistency_score"]
    ]
    print(json.dumps(top.to_dict(orient="records"), indent=2, sort_keys=True))


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    main()
