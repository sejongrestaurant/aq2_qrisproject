"""Generate parameter grids for MUST30 backtest experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Literal

FactorSetting = Literal["balanced", "momentum_focused", "quality_focused", "low_volatility_focused"]
WeightMethod = Literal["equal_weight", "score_weight", "rank_weight"]
RebalanceFrequency = Literal["monthly", "quarterly"]

DEFAULT_TRAIN_PERIOD = ("2014-01-01", "2020-12-31")
DEFAULT_VALIDATION_PERIOD = ("2021-01-01", "2023-12-31")
DEFAULT_TEST_PERIOD = ("2024-01-01", None)

FACTOR_SETTINGS: dict[FactorSetting, dict[str, float]] = {
    "balanced": {
        "momentum": 0.25,
        "relative_strength": 0.15,
        "quality": 0.20,
        "growth": 0.20,
        "low_volatility": 0.15,
        "liquidity": 0.05,
    },
    "momentum_focused": {
        "momentum": 0.40,
        "relative_strength": 0.20,
        "quality": 0.15,
        "growth": 0.10,
        "low_volatility": 0.10,
        "liquidity": 0.05,
    },
    "quality_focused": {
        "momentum": 0.15,
        "relative_strength": 0.10,
        "quality": 0.40,
        "growth": 0.15,
        "low_volatility": 0.15,
        "liquidity": 0.05,
    },
    "low_volatility_focused": {
        "momentum": 0.15,
        "relative_strength": 0.10,
        "quality": 0.20,
        "growth": 0.10,
        "low_volatility": 0.40,
        "liquidity": 0.05,
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment parameter combination."""

    experiment_id: str
    target_size: int
    rebalance_frequency: RebalanceFrequency
    weight_method: WeightMethod
    regime_equity_weights: dict[str, float]
    transaction_cost: float
    factor_setting: FactorSetting
    factor_weights: dict[str, float]
    train_period: tuple[str, str | None] = DEFAULT_TRAIN_PERIOD
    validation_period: tuple[str, str | None] = DEFAULT_VALIDATION_PERIOD
    test_period: tuple[str, str | None] = DEFAULT_TEST_PERIOD
    metadata: dict[str, str] = field(
        default_factory=lambda: {
            "test_warning": (
                "Do not modify parameters based on test-period results; use test only once for final evaluation."
            )
        }
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return asdict(self)


def generate_experiment_configs(
    *,
    target_sizes: tuple[int, ...] = (20, 30, 40),
    rebalance_frequencies: tuple[RebalanceFrequency, ...] = ("monthly", "quarterly"),
    weight_methods: tuple[WeightMethod, ...] = ("equal_weight", "score_weight", "rank_weight"),
    regime_weight_sets: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.8, 0.5),
        (1.0, 0.7, 0.3),
        (0.9, 0.7, 0.4),
    ),
    transaction_costs: tuple[float, ...] = (0.001, 0.002, 0.003),
    factor_settings: tuple[FactorSetting, ...] = (
        "balanced",
        "momentum_focused",
        "quality_focused",
        "low_volatility_focused",
    ),
) -> list[ExperimentConfig]:
    """Generate the full experiment grid with deterministic experiment IDs."""
    configs = []
    for values in product(
        target_sizes,
        rebalance_frequencies,
        weight_methods,
        regime_weight_sets,
        transaction_costs,
        factor_settings,
    ):
        target_size, frequency, method, regime_tuple, cost, factor_setting = values
        regime_weights = {
            "Risk-On": regime_tuple[0],
            "Neutral": regime_tuple[1],
            "Risk-Off": regime_tuple[2],
        }
        payload = {
            "target_size": target_size,
            "rebalance_frequency": frequency,
            "weight_method": method,
            "regime_equity_weights": regime_weights,
            "transaction_cost": cost,
            "factor_setting": factor_setting,
        }
        experiment_id = _experiment_id(payload)
        configs.append(
            ExperimentConfig(
                experiment_id=experiment_id,
                target_size=int(target_size),
                rebalance_frequency=frequency,
                weight_method=method,
                regime_equity_weights=regime_weights,
                transaction_cost=float(cost),
                factor_setting=factor_setting,
                factor_weights=FACTOR_SETTINGS[factor_setting].copy(),
            )
        )
    return sorted(configs, key=lambda item: item.experiment_id)


def _experiment_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return f"exp_{digest}"


__all__ = [
    "DEFAULT_TEST_PERIOD",
    "DEFAULT_TRAIN_PERIOD",
    "DEFAULT_VALIDATION_PERIOD",
    "ExperimentConfig",
    "FACTOR_SETTINGS",
    "FactorSetting",
    "RebalanceFrequency",
    "WeightMethod",
    "generate_experiment_configs",
]
