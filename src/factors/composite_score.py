"""Composite factor scoring and analysis utilities."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.config.factor_config import FactorConfig, FactorName
from src.database.repositories import upsert_factor_score
from src.factors.common import cross_sectional_zscore, winsorize_series

FACTOR_RAW_COLUMNS: dict[FactorName, str] = {
    "momentum": "momentum_raw",
    "relative_strength": "relative_strength_raw",
    "quality": "quality_raw",
    "growth": "growth_raw",
    "low_volatility": "low_volatility_raw",
    "liquidity": "liquidity_raw",
}
FACTOR_SCORE_COLUMNS: dict[FactorName, str] = {
    "momentum": "momentum_score",
    "relative_strength": "relative_strength_score",
    "quality": "quality_score",
    "growth": "growth_score",
    "low_volatility": "low_volatility_score",
    "liquidity": "liquidity_score",
}
INVERTED_FACTORS: frozenset[FactorName] = frozenset({"low_volatility"})


def calculate_composite_scores(
    factor_df: pd.DataFrame,
    config: FactorConfig | None = None,
) -> pd.DataFrame:
    """Calculate composite factor scores and ranks.

    Input contract:
    - Required columns: `calculation_date`, `ticker`, and six raw factor columns:
      `momentum_raw`, `relative_strength_raw`, `quality_raw`, `growth_raw`,
      `low_volatility_raw`, and `liquidity_raw`.
    - Optional `sector` enables sector-neutral scoring. Missing sectors become `Unknown`.
    - Optional `available_date` is treated as a point-in-time guard. Rows with
      `available_date > calculation_date` are excluded before scoring.

    Scoring:
    - Each raw factor is winsorized cross-sectionally by `calculation_date`, then converted to a
      z-score. In `sector_neutral` mode, winsorization and z-score are applied within
      `calculation_date + sector` groups.
    - `low_volatility_raw` is directionally inverted before z-scoring so lower volatility receives
      a higher score.
    - Missing policies:
      `exclude` requires all six factor scores.
      `available_weight_rescale` uses available scores only, rescales their weights to sum to one,
      and requires `min_available_factors` scores.
      `median_impute` fills missing factor scores with the calculation-date median, then applies
      the configured weights.

    Ranking:
    - `universe_rank` is sorted by `composite_score` descending.
    - Ties are broken deterministically by `ticker` ascending, so reruns produce identical ranks.
    """
    resolved_config = config or FactorConfig()
    resolved_config.validate()
    normalized = _normalize_factor_frame(factor_df)
    scored = _add_factor_scores(normalized, resolved_config)
    scored = _add_composite_score(scored, resolved_config)
    scored = scored.dropna(subset=["composite_score"]).copy()
    scored = _add_rank(scored)
    return scored.reset_index(drop=True)


def upsert_composite_scores(session: Session, composite_df: pd.DataFrame) -> int:
    """Upsert composite score rows into `factor_scores` and return affected row count."""
    required_columns = {
        "calculation_date",
        "ticker",
        "composite_score",
        "universe_rank",
        *FACTOR_RAW_COLUMNS.values(),
        *FACTOR_SCORE_COLUMNS.values(),
    }
    missing_columns = required_columns - set(composite_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns for factor_scores upsert: {', '.join(sorted(missing_columns))}"
        )

    for row in composite_df.itertuples(index=False):
        values = {
            "calculation_date": _date_value(row.calculation_date),
            "ticker": str(row.ticker).zfill(6),
            "composite_score": float(row.composite_score),
            "universe_rank": int(row.universe_rank),
        }
        for column in FACTOR_RAW_COLUMNS.values():
            values[column] = _nullable_float(getattr(row, column))
        for column in FACTOR_SCORE_COLUMNS.values():
            values[column] = _nullable_float(getattr(row, column))
        upsert_factor_score(session, values)
    return len(composite_df)


def calculate_factor_contributions(
    composite_df: pd.DataFrame,
    config: FactorConfig | None = None,
) -> pd.DataFrame:
    """Return weighted per-factor contributions to each composite score."""
    resolved_config = config or FactorConfig()
    resolved_config.validate()
    result = composite_df.loc[:, ["calculation_date", "ticker"]].copy()
    for factor, score_column in FACTOR_SCORE_COLUMNS.items():
        contribution_column = f"{factor}_contribution"
        result[contribution_column] = composite_df[score_column] * resolved_config.weights[factor]
    return result


def calculate_sector_average_scores(composite_df: pd.DataFrame) -> pd.DataFrame:
    """Return sector-level average composite and factor scores."""
    if "sector" not in composite_df.columns:
        raise ValueError("sector column is required for sector average scores")
    score_columns = ["composite_score", *FACTOR_SCORE_COLUMNS.values()]
    return (
        composite_df.groupby(["calculation_date", "sector"], as_index=False)[score_columns]
        .mean()
        .sort_values(["calculation_date", "sector"])
    )


def calculate_top_n_contributions(
    composite_df: pd.DataFrame,
    config: FactorConfig | None = None,
    *,
    n: int = 30,
) -> pd.DataFrame:
    """Return factor contributions for the top N names by calculation date."""
    top = composite_df.loc[composite_df["universe_rank"] <= n].copy()
    return calculate_factor_contributions(top, config)


def calculate_factor_correlation(composite_df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix of factor score columns."""
    return composite_df.loc[:, list(FACTOR_SCORE_COLUMNS.values())].corr()


def _normalize_factor_frame(factor_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"calculation_date", "ticker", *FACTOR_RAW_COLUMNS.values()}
    missing_columns = required_columns - set(factor_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required factor columns: {', '.join(sorted(missing_columns))}")

    normalized = factor_df.copy()
    normalized["calculation_date"] = pd.to_datetime(normalized["calculation_date"])
    normalized["ticker"] = normalized["ticker"].astype(str).str.zfill(6)
    if "available_date" in normalized.columns:
        normalized["available_date"] = pd.to_datetime(normalized["available_date"])
        normalized = normalized.loc[normalized["available_date"] <= normalized["calculation_date"]]
    if "sector" not in normalized.columns:
        normalized["sector"] = "Unknown"
    normalized["sector"] = normalized["sector"].fillna("Unknown").astype(str)
    for column in FACTOR_RAW_COLUMNS.values():
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.replace([np.inf, -np.inf], np.nan)


def _add_factor_scores(df: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    result = df.copy()
    group_columns = (
        ["calculation_date", "sector"]
        if config.scoring_mode == "sector_neutral"
        else ["calculation_date"]
    )
    for factor, raw_column in FACTOR_RAW_COLUMNS.items():
        values = -result[raw_column] if factor in INVERTED_FACTORS else result[raw_column]
        score_column = FACTOR_SCORE_COLUMNS[factor]
        result[score_column] = values.groupby(
            [result[column] for column in group_columns]
        ).transform(
            lambda group: cross_sectional_zscore(
                winsorize_series(
                    group,
                    config.winsorize_lower_quantile,
                    config.winsorize_upper_quantile,
                )
            )
        )
    return result


def _add_composite_score(df: pd.DataFrame, config: FactorConfig) -> pd.DataFrame:
    result = df.copy()
    score_columns = list(FACTOR_SCORE_COLUMNS.values())

    if config.missing_policy == "median_impute":
        for score_column in score_columns:
            result[score_column] = result.groupby("calculation_date")[score_column].transform(
                lambda group: group.fillna(group.median())
            )
        result["composite_score"] = _weighted_sum(result, config.weights)
        return result

    available_count = result[score_columns].notna().sum(axis=1)
    if config.missing_policy == "exclude":
        result = result.loc[available_count == len(score_columns)].copy()
        result["composite_score"] = _weighted_sum(result, config.weights)
        return result

    result = result.loc[available_count >= config.min_available_factors].copy()
    weighted_sum = pd.Series(0.0, index=result.index)
    available_weight = pd.Series(0.0, index=result.index)
    for factor, score_column in FACTOR_SCORE_COLUMNS.items():
        score = result[score_column]
        present = score.notna()
        weighted_sum.loc[present] += score.loc[present] * config.weights[factor]
        available_weight.loc[present] += config.weights[factor]
    result["composite_score"] = weighted_sum / available_weight.replace(0.0, np.nan)
    return result


def _weighted_sum(df: pd.DataFrame, weights: dict[FactorName, float]) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for factor, score_column in FACTOR_SCORE_COLUMNS.items():
        total += df[score_column] * weights[factor]
    return total


def _add_rank(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(
        ["calculation_date", "composite_score", "ticker"], ascending=[True, False, True]
    ).copy()
    result["universe_rank"] = result.groupby("calculation_date").cumcount() + 1
    return result


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _date_value(value: object) -> date:
    return pd.Timestamp(value).date()


__all__ = [
    "FACTOR_RAW_COLUMNS",
    "FACTOR_SCORE_COLUMNS",
    "calculate_composite_scores",
    "calculate_factor_contributions",
    "calculate_factor_correlation",
    "calculate_sector_average_scores",
    "calculate_top_n_contributions",
    "upsert_composite_scores",
]
