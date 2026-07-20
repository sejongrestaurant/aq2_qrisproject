"""Portfolio selection constraints and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

RelaxationStep = Literal[
    "sector_max",
    "kosdaq_max",
    "min_core",
    "min_defensive",
    "liquidity",
    "zero_volume",
]


@dataclass(frozen=True)
class PortfolioConstraints:
    """Configurable constraints for monthly top-name portfolio selection."""

    target_size: int = 30
    max_sector_count: int = 6
    max_kosdaq_count: int = 10
    min_core_count: int = 5
    min_defensive_count: int = 3
    min_listing_trading_days: int = 252
    min_avg_trading_value_20d: float = 0.0
    max_zero_volume_ratio_60d: float = 0.10
    min_available_factors: int = 4
    relaxation_order: tuple[RelaxationStep, ...] = ("sector_max", "kosdaq_max")

    def relaxed(self, steps: set[RelaxationStep]) -> PortfolioConstraints:
        """Return a copy with the requested soft constraints relaxed."""
        updated = self
        if "sector_max" in steps:
            updated = replace(updated, max_sector_count=updated.target_size)
        if "kosdaq_max" in steps:
            updated = replace(updated, max_kosdaq_count=updated.target_size)
        if "min_core" in steps:
            updated = replace(updated, min_core_count=0)
        if "min_defensive" in steps:
            updated = replace(updated, min_defensive_count=0)
        if "liquidity" in steps:
            updated = replace(updated, min_avg_trading_value_20d=0.0)
        if "zero_volume" in steps:
            updated = replace(updated, max_zero_volume_ratio_60d=1.0)
        return updated


def summarize_constraints(
    selected: pd.DataFrame,
    excluded: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    relaxed_steps: set[RelaxationStep] | None = None,
) -> dict[str, object]:
    """Build a deterministic constraint summary for selection results."""
    relaxed_steps = relaxed_steps or set()
    role_counts = _counts(selected, "universe_role")
    sector_counts = _counts(selected, "sector")
    market_counts = _counts(selected, "market")
    shortage = max(constraints.target_size - len(selected), 0)

    return {
        "target_size": constraints.target_size,
        "selected_count": int(len(selected)),
        "shortage": int(shortage),
        "sector_counts": sector_counts,
        "market_counts": market_counts,
        "role_counts": role_counts,
        "kosdaq_count": int(market_counts.get("KOSDAQ", 0)),
        "core_count": int(role_counts.get("Core", 0)),
        "defensive_count": int(role_counts.get("Defensive", 0)),
        "constraints": {
            "max_sector_count": constraints.max_sector_count,
            "max_kosdaq_count": constraints.max_kosdaq_count,
            "min_core_count": constraints.min_core_count,
            "min_defensive_count": constraints.min_defensive_count,
            "min_listing_trading_days": constraints.min_listing_trading_days,
            "min_avg_trading_value_20d": constraints.min_avg_trading_value_20d,
            "max_zero_volume_ratio_60d": constraints.max_zero_volume_ratio_60d,
            "min_available_factors": constraints.min_available_factors,
        },
        "relaxed_steps": sorted(relaxed_steps),
        "exclusion_reason_counts": _counts(excluded, "exclusion_reason"),
        "shortage_reasons": shortage_reasons(selected, excluded, constraints),
    }


def shortage_reasons(
    selected: pd.DataFrame,
    excluded: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> list[str]:
    """Explain which constraints most likely caused an underfilled result."""
    if len(selected) >= constraints.target_size:
        return []

    reasons: list[str] = []
    if excluded.empty or "exclusion_reason" not in excluded.columns:
        return ["eligible universe smaller than target_size"]

    reason_counts = excluded["exclusion_reason"].value_counts().sort_index()
    for reason, count in reason_counts.items():
        reasons.append(f"{reason}: {int(count)}")
    if not reasons:
        reasons.append("eligible universe smaller than target_size")
    return reasons


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts().sort_index().items()}


__all__ = [
    "PortfolioConstraints",
    "RelaxationStep",
    "shortage_reasons",
    "summarize_constraints",
]
