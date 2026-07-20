"""Configurable factor weights and composite-score policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FactorName = Literal[
    "momentum",
    "relative_strength",
    "quality",
    "growth",
    "low_volatility",
    "liquidity",
]
MissingFactorPolicy = Literal["exclude", "available_weight_rescale", "median_impute"]
ScoringMode = Literal["universe", "sector_neutral"]

DEFAULT_FACTOR_WEIGHTS: dict[FactorName, float] = {
    "momentum": 0.25,
    "relative_strength": 0.15,
    "quality": 0.20,
    "growth": 0.20,
    "low_volatility": 0.15,
    "liquidity": 0.05,
}


@dataclass(frozen=True)
class FactorConfig:
    """Composite factor configuration.

    The weights must sum to 1.0. The default missing-data policy rescales the available factor
    weights but requires at least four non-null factor scores per stock.
    """

    weights: dict[FactorName, float] = field(default_factory=lambda: DEFAULT_FACTOR_WEIGHTS.copy())
    missing_policy: MissingFactorPolicy = "available_weight_rescale"
    min_available_factors: int = 4
    scoring_mode: ScoringMode = "universe"
    winsorize_lower_quantile: float = 0.05
    winsorize_upper_quantile: float = 0.95

    def validate(self) -> None:
        """Validate factor weights and policy settings."""
        missing_weights = set(DEFAULT_FACTOR_WEIGHTS) - set(self.weights)
        extra_weights = set(self.weights) - set(DEFAULT_FACTOR_WEIGHTS)
        if missing_weights or extra_weights:
            raise ValueError(
                "Factor weights must contain exactly these keys: "
                f"{', '.join(DEFAULT_FACTOR_WEIGHTS)}"
            )

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"Factor weights must sum to 1.0; got {weight_sum:.12f}")

        if self.min_available_factors < 1 or self.min_available_factors > len(self.weights):
            raise ValueError("min_available_factors must be between 1 and the number of factors")

        if self.winsorize_lower_quantile < 0 or self.winsorize_upper_quantile > 1:
            raise ValueError("winsorization quantiles must be inside [0, 1]")

        if self.winsorize_lower_quantile > self.winsorize_upper_quantile:
            raise ValueError("lower winsorization quantile must be <= upper quantile")


__all__ = [
    "DEFAULT_FACTOR_WEIGHTS",
    "FactorConfig",
    "FactorName",
    "MissingFactorPolicy",
    "ScoringMode",
]
