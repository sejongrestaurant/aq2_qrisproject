"""Portfolio weight optimization with practical ETF constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

WeightMethod = Literal["equal_weight", "score_weight", "rank_weight"]
LowerBoundPolicy = Literal["floor", "drop"]


class PortfolioOptimizationError(RuntimeError):
    """Raised when portfolio constraints cannot be satisfied."""


@dataclass(frozen=True)
class WeightConstraints:
    """Constraints used by the portfolio weight optimizer."""

    target_size: int = 30
    max_stock_weight: float = 0.07
    min_stock_weight: float = 0.01
    max_sector_weight: float = 0.25
    max_kosdaq_weight: float = 0.35
    lower_bound_policy: LowerBoundPolicy = "floor"
    max_iterations: int = 100
    tolerance: float = 1e-12

    def validate(self) -> None:
        """Validate optimizer constraints."""
        if self.target_size < 1:
            raise ValueError("target_size must be positive")
        if self.min_stock_weight < 0.0:
            raise ValueError("min_stock_weight must be non-negative")
        if self.max_stock_weight <= 0.0:
            raise ValueError("max_stock_weight must be positive")
        if self.min_stock_weight > self.max_stock_weight:
            raise ValueError("min_stock_weight must be <= max_stock_weight")
        if self.max_sector_weight <= 0.0 or self.max_kosdaq_weight <= 0.0:
            raise ValueError("sector and KOSDAQ caps must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


def optimize_portfolio_weights(
    selected_portfolio: pd.DataFrame,
    *,
    method: WeightMethod = "score_weight",
    equity_weight: float = 1.0,
    constraints: WeightConstraints | None = None,
) -> pd.DataFrame:
    """Calculate constrained stock and cash portfolio weights."""
    resolved = constraints or WeightConstraints()
    resolved.validate()
    if equity_weight < 0.0 or equity_weight > 1.0:
        raise ValueError("equity_weight must be between 0.0 and 1.0")

    candidates = _prepare_candidates(selected_portfolio, resolved)
    preferences = _preference_scores(candidates, method)
    active_candidates, stock_weights = _solve_with_lower_policy(
        candidates,
        preferences,
        equity_weight,
        resolved,
    )
    result = _build_result(active_candidates, stock_weights, method, equity_weight)
    errors = validate_portfolio_weights(result, resolved)
    if errors:
        raise PortfolioOptimizationError("; ".join(errors))
    return result


def validate_portfolio_weights(
    weights: pd.DataFrame,
    constraints: WeightConstraints | None = None,
) -> list[str]:
    """Validate portfolio weights and return human-readable errors."""
    resolved = constraints or WeightConstraints()
    resolved.validate()
    required_columns = {"ticker", "target_weight"}
    missing = required_columns - set(weights.columns)
    if missing:
        return [f"Missing weight columns: {', '.join(sorted(missing))}"]

    errors: list[str] = []
    normalized = weights.copy()
    normalized["target_weight"] = pd.to_numeric(normalized["target_weight"], errors="coerce")
    if normalized["target_weight"].isna().any():
        errors.append("target_weight contains missing or non-numeric values")
        return errors

    total_weight = float(normalized["target_weight"].sum())
    if abs(total_weight - 1.0) > resolved.tolerance * 10:
        errors.append(f"total weight must equal 1.0; got {total_weight:.12f}")

    stocks = normalized.loc[normalized["ticker"] != "CASH"].copy()
    if len(stocks) != resolved.target_size:
        errors.append(f"stock count must equal {resolved.target_size}; got {len(stocks)}")

    overweight = stocks.loc[
        stocks["target_weight"] > resolved.max_stock_weight + resolved.tolerance
    ]
    if not overweight.empty:
        errors.append(
            f"stock max weight exceeded: {', '.join(overweight['ticker'].astype(str).tolist())}"
        )

    underweight = stocks.loc[
        stocks["target_weight"] < resolved.min_stock_weight - resolved.tolerance
    ]
    if not underweight.empty:
        errors.append(
            f"stock min weight violated: {', '.join(underweight['ticker'].astype(str).tolist())}"
        )

    if "sector" in stocks.columns:
        sector_weights = stocks.groupby("sector")["target_weight"].sum()
        exceeded = sector_weights.loc[
            sector_weights > resolved.max_sector_weight + resolved.tolerance
        ]
        if not exceeded.empty:
            errors.append(
                f"sector max weight exceeded: {', '.join(map(str, exceeded.index.tolist()))}"
            )

    if "market" in stocks.columns:
        kosdaq_weight = float(stocks.loc[stocks["market"] == "KOSDAQ", "target_weight"].sum())
        if kosdaq_weight > resolved.max_kosdaq_weight + resolved.tolerance:
            errors.append(f"KOSDAQ max weight exceeded: {kosdaq_weight:.12f}")

    cash = normalized.loc[normalized["ticker"] == "CASH"]
    if len(cash) != 1:
        errors.append("portfolio must contain exactly one CASH row")
    return errors


def _prepare_candidates(
    selected_portfolio: pd.DataFrame, constraints: WeightConstraints
) -> pd.DataFrame:
    required_columns = {"ticker", "composite_score", "sector", "market"}
    missing = required_columns - set(selected_portfolio.columns)
    if missing:
        raise ValueError(f"Missing selected portfolio columns: {', '.join(sorted(missing))}")
    if len(selected_portfolio) != constraints.target_size:
        raise ValueError(f"selected_portfolio must contain {constraints.target_size} stocks")

    result = selected_portfolio.copy()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["composite_score"] = pd.to_numeric(result["composite_score"], errors="coerce")
    if result["composite_score"].isna().any():
        raise ValueError("composite_score contains missing or non-numeric values")
    if "rank" not in result.columns:
        result = result.sort_values(["composite_score", "ticker"], ascending=[False, True]).copy()
        result["rank"] = range(1, len(result) + 1)
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype(int)
    return result.sort_values(["rank", "ticker"]).reset_index(drop=True)


def _preference_scores(candidates: pd.DataFrame, method: WeightMethod) -> pd.Series:
    if method == "equal_weight":
        return pd.Series(1.0, index=candidates.index)
    if method == "score_weight":
        scores = candidates["composite_score"].astype(float)
        shifted = scores - scores.min() + 1e-9
        if float(shifted.sum()) <= 0.0:
            return pd.Series(1.0, index=candidates.index)
        return shifted
    if method == "rank_weight":
        ranks = candidates["rank"].astype(float)
        return (len(candidates) - ranks + 1).clip(lower=1.0)
    raise ValueError(f"Unknown weighting method: {method}")


def _solve_with_lower_policy(
    candidates: pd.DataFrame,
    preferences: pd.Series,
    equity_weight: float,
    constraints: WeightConstraints,
) -> tuple[pd.DataFrame, pd.Series]:
    if constraints.lower_bound_policy == "floor":
        weights = _allocate_with_caps(
            candidates, preferences, equity_weight, constraints, use_floor=True
        )
        return candidates, weights

    active = candidates.copy()
    active_preferences = preferences.copy()
    while True:
        weights = _allocate_with_caps(
            active, active_preferences, equity_weight, constraints, use_floor=False
        )
        under = weights.loc[weights < constraints.min_stock_weight - constraints.tolerance]
        if under.empty:
            return active, weights
        if len(under) == len(weights):
            raise PortfolioOptimizationError("lower_bound_policy=drop removed every stock")
        active = active.drop(index=under.index)
        active_preferences = active_preferences.drop(index=under.index)


def _allocate_with_caps(
    candidates: pd.DataFrame,
    preferences: pd.Series,
    equity_weight: float,
    constraints: WeightConstraints,
    *,
    use_floor: bool,
) -> pd.Series:
    lower = constraints.min_stock_weight if use_floor else 0.0
    weights = pd.Series(lower, index=candidates.index, dtype="float64")
    remaining = equity_weight - float(weights.sum())
    if remaining < -constraints.tolerance:
        raise PortfolioOptimizationError("minimum stock weights exceed equity allocation")
    if abs(remaining) <= constraints.tolerance:
        return weights

    eligible = set(candidates.index)
    for _ in range(constraints.max_iterations):
        if remaining <= constraints.tolerance:
            break
        capacities = _remaining_capacities(candidates, weights, constraints)
        eligible = {idx for idx in eligible if capacities.loc[idx] > constraints.tolerance}
        if not eligible:
            raise PortfolioOptimizationError(
                "portfolio constraints leave no capacity for redistribution"
            )

        pref = preferences.loc[list(eligible)].astype(float)
        if float(pref.sum()) <= 0.0:
            pref = pd.Series(1.0, index=pref.index)
        increments = remaining * pref / float(pref.sum())
        increments = increments.clip(upper=capacities.loc[increments.index])
        increments = _cap_group_increments(candidates, weights, increments, constraints)
        weights.loc[increments.index] += increments
        distributed = float(increments.sum())
        if distributed <= constraints.tolerance:
            raise PortfolioOptimizationError("redistribution failed to make progress")
        remaining -= distributed
    else:
        raise PortfolioOptimizationError("portfolio weight redistribution did not converge")

    if abs(remaining) > constraints.tolerance:
        raise PortfolioOptimizationError("portfolio weight redistribution did not converge")
    return weights


def _cap_group_increments(
    candidates: pd.DataFrame,
    weights: pd.Series,
    increments: pd.Series,
    constraints: WeightConstraints,
) -> pd.Series:
    capped = increments.copy()
    for _sector, sector_indices in candidates.groupby("sector").groups.items():
        index = capped.index.intersection(sector_indices)
        if index.empty:
            continue
        remaining = constraints.max_sector_weight - float(weights.loc[sector_indices].sum())
        capped.loc[index] = _scale_to_capacity(capped.loc[index], remaining)

    kosdaq_indices = candidates.index[candidates["market"] == "KOSDAQ"]
    index = capped.index.intersection(kosdaq_indices)
    if not index.empty:
        remaining = constraints.max_kosdaq_weight - float(weights.loc[kosdaq_indices].sum())
        capped.loc[index] = _scale_to_capacity(capped.loc[index], remaining)
    return capped


def _scale_to_capacity(values: pd.Series, capacity: float) -> pd.Series:
    if capacity <= 0.0:
        return pd.Series(0.0, index=values.index)
    total = float(values.sum())
    if total <= capacity:
        return values
    return values * (capacity / total)


def _remaining_capacities(
    candidates: pd.DataFrame,
    weights: pd.Series,
    constraints: WeightConstraints,
) -> pd.Series:
    capacities = (
        pd.Series(constraints.max_stock_weight, index=candidates.index, dtype="float64") - weights
    )

    sector_sums = weights.groupby(candidates["sector"]).sum()
    for sector, total in sector_sums.items():
        sector_capacity = constraints.max_sector_weight - float(total)
        sector_mask = candidates["sector"] == sector
        capacities.loc[sector_mask] = capacities.loc[sector_mask].clip(
            upper=max(sector_capacity, 0.0)
        )

    kosdaq_mask = candidates["market"] == "KOSDAQ"
    kosdaq_total = float(weights.loc[kosdaq_mask].sum())
    kosdaq_capacity = constraints.max_kosdaq_weight - kosdaq_total
    capacities.loc[kosdaq_mask] = capacities.loc[kosdaq_mask].clip(upper=max(kosdaq_capacity, 0.0))
    return capacities.clip(lower=0.0)


def _build_result(
    candidates: pd.DataFrame,
    stock_weights: pd.Series,
    method: WeightMethod,
    equity_weight: float,
) -> pd.DataFrame:
    result = candidates.copy()
    result["target_weight"] = stock_weights.reindex(result.index).fillna(0.0).astype(float)
    result["weight_method"] = method
    result["weight_reason"] = result.apply(
        lambda row: (
            f"{method}: composite_score={float(row['composite_score']):.6f}; "
            f"rank={int(row['rank'])}; equity_weight={equity_weight:.6f}; constraints_applied"
        ),
        axis=1,
    )
    stock_sum = float(result["target_weight"].sum())
    cash_weight = 1.0 - stock_sum
    cash = pd.DataFrame(
        [
            {
                "ticker": "CASH",
                "sector": "Cash",
                "market": "Cash",
                "composite_score": pd.NA,
                "rank": len(result) + 1,
                "target_weight": cash_weight,
                "weight_method": method,
                "weight_reason": f"cash_residual: 1 - stock_weight_sum({stock_sum:.12f})",
            }
        ]
    )
    return pd.concat([result, cash], ignore_index=True)


__all__ = [
    "LowerBoundPolicy",
    "PortfolioOptimizationError",
    "WeightConstraints",
    "WeightMethod",
    "optimize_portfolio_weights",
    "validate_portfolio_weights",
]
