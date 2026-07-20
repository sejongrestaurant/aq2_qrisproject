"""Analyze and rank experiment results."""

from __future__ import annotations

import pandas as pd

METRIC_WEIGHTS: dict[str, float] = {
    "cagr": 0.25,
    "sharpe_ratio": 0.25,
    "maximum_drawdown": 0.20,
    "calmar_ratio": 0.15,
    "annual_turnover": 0.075,
    "total_transaction_cost": 0.075,
}


def analyze_experiments(results: pd.DataFrame) -> pd.DataFrame:
    """Return a comparison table with multi-criteria scores.

    The score intentionally combines CAGR with risk, drawdown, turnover, and cost metrics so the
    top strategy is not selected from CAGR alone.
    """
    if results.empty:
        return results.copy()
    completed = results.loc[results["status"] == "completed"].copy()
    failed = results.loc[results["status"] != "completed"].copy()
    if completed.empty:
        output = failed.copy()
        output["multi_criteria_score"] = pd.NA
        output["consistency_score"] = pd.NA
        output["is_top_strategy"] = False
        output["is_consistent_strategy"] = False
        return output.sort_values("experiment_id").reset_index(drop=True)

    scored = completed.copy()
    scored["multi_criteria_score"] = 0.0
    scored["consistency_score"] = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        if metric not in scored.columns:
            continue
        higher_is_better = metric not in {
            "maximum_drawdown",
            "annual_turnover",
            "total_transaction_cost",
        }
        normalized = _normalize(scored[metric].astype(float), higher_is_better=higher_is_better)
        scored["multi_criteria_score"] += normalized * weight

    if {"train_multi_score", "validation_multi_score", "test_multi_score"} <= set(scored.columns):
        period_scores = scored[["train_multi_score", "validation_multi_score", "test_multi_score"]]
        scored["consistency_score"] = period_scores.mean(axis=1) - period_scores.std(axis=1).fillna(
            0.0
        )
    else:
        grouped = scored.groupby("experiment_id")["multi_criteria_score"]
        scored["consistency_score"] = grouped.transform("mean") - grouped.transform("std").fillna(
            0.0
        )

    scored["is_top_strategy"] = (
        scored["multi_criteria_score"] == scored["multi_criteria_score"].max()
    )
    consistency_cutoff = scored["consistency_score"].quantile(0.90)
    scored["is_consistent_strategy"] = scored["consistency_score"] >= consistency_cutoff
    scored = scored.sort_values(
        ["multi_criteria_score", "consistency_score", "experiment_id"],
        ascending=[False, False, True],
    )
    if failed.empty:
        return scored.reset_index(drop=True)
    failed = failed.copy()
    failed["multi_criteria_score"] = pd.NA
    failed["consistency_score"] = pd.NA
    failed["is_top_strategy"] = False
    failed["is_consistent_strategy"] = False
    return pd.concat([scored, failed], ignore_index=True, sort=False).reset_index(drop=True)


def select_consistent_strategies(results: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """Return strategies with the highest period consistency."""
    analyzed = analyze_experiments(results)
    if analyzed.empty:
        return analyzed
    return analyzed.sort_values(
        ["consistency_score", "multi_criteria_score"],
        ascending=[False, False],
    ).head(top_n)


def _normalize(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = series.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    if values.max() == values.min():
        return pd.Series(0.5, index=series.index)
    scaled = (values - values.min()) / (values.max() - values.min())
    return scaled if higher_is_better else 1.0 - scaled


__all__ = ["METRIC_WEIGHTS", "analyze_experiments", "select_consistent_strategies"]
