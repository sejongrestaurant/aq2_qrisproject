"""CLI entrypoint for running MUST30 backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestConfig, run_backtest
from src.universe.loader import load_universe


def main(argv: list[str] | None = None) -> None:
    """Run the backtest CLI."""
    parser = argparse.ArgumentParser(description="Run a MUST30 backtest")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--strategy",
        choices=["score_weight", "equal_weight", "rank_weight"],
        default="score_weight",
    )
    parser.add_argument("--transaction-cost", type=float, default=0.002)
    parser.add_argument("--prices", type=Path, default=Path("data/prices.parquet"))
    parser.add_argument(
        "--universe", type=Path, default=Path("data/universe/korea_active_etf_universe_100.csv")
    )
    parser.add_argument("--factor-scores", type=Path, default=Path("data/factor_scores.parquet"))
    parser.add_argument("--market-data", type=Path, default=Path("data/market_data.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backtest"))
    parser.add_argument("--target-size", type=int, default=30)
    args = parser.parse_args(argv)

    prices = _read_table(args.prices)
    universe = (
        load_universe(args.universe)
        if args.universe.suffix.lower() == ".csv"
        else _read_table(args.universe)
    )
    factor_scores = _read_table(args.factor_scores)
    market_data = _read_table(args.market_data)
    result = run_backtest(
        prices,
        universe,
        factor_scores,
        market_data,
        config=BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
            transaction_cost=args.transaction_cost,
            target_size=args.target_size,
        ),
        output_dir=args.output_dir,
    )
    print(json.dumps(result.performance_summary, indent=2, sort_keys=True))


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


if __name__ == "__main__":
    main()
