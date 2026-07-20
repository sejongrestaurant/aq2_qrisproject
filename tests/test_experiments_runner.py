"""Tests for experiment execution helpers."""

from __future__ import annotations

import pandas as pd

from src.experiments.config_generator import FACTOR_SETTINGS, generate_experiment_configs
from src.experiments.runner import _backtest_config_for_experiment, _factor_scores_for_setting


def test_experiment_transaction_cost_updates_commission_and_impact() -> None:
    """Experiment cost candidates should drive actual backtest execution costs."""
    config = next(
        item for item in generate_experiment_configs() if item.transaction_cost == 0.003
    )

    backtest_config = _backtest_config_for_experiment(
        config,
        start_date="2024-01-01",
        end_date="2024-12-31",
        strategy_name="test",
    )

    assert backtest_config.transaction_cost == 0.003
    assert abs(backtest_config.commission - 0.00225) <= 1e-12
    assert abs(backtest_config.market_impact - 0.00075) <= 1e-12
    assert abs(backtest_config.commission + backtest_config.market_impact - 0.003) <= 1e-12


def test_factor_scores_are_recomputed_for_experiment_weights() -> None:
    """Raw factor inputs should be rescored with the selected experiment factor weights."""
    raw = _raw_factor_scores()

    balanced = _factor_scores_for_setting(
        raw,
        "balanced",
        factor_weights=FACTOR_SETTINGS["balanced"],
    )
    momentum = _factor_scores_for_setting(
        raw,
        "momentum_focused",
        factor_weights=FACTOR_SETTINGS["momentum_focused"],
    )

    balanced_scores = balanced.set_index("ticker")["composite_score"]
    momentum_scores = momentum.set_index("ticker")["composite_score"]
    assert not balanced_scores.equals(momentum_scores)
    assert momentum["factor_setting"].eq("momentum_focused").all()


def test_precomputed_factor_setting_is_used_when_available() -> None:
    """Precomputed setting variants should not be overwritten by raw rescoring."""
    scores = pd.DataFrame(
        {
            "calculation_date": [pd.Timestamp("2024-06-30")] * 2,
            "ticker": ["000001", "000002"],
            "composite_score": [1.0, 2.0],
            "factor_setting": ["balanced", "momentum_focused"],
        }
    )

    result = _factor_scores_for_setting(
        scores,
        "momentum_focused",
        factor_weights=FACTOR_SETTINGS["momentum_focused"],
    )

    assert result["ticker"].tolist() == ["000002"]
    assert result["composite_score"].tolist() == [2.0]


def _raw_factor_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _raw_factor_row("000001", 1.0, 0.1, 0.1, 0.1, 0.30, 10.0),
            _raw_factor_row("000002", 0.1, 1.0, 0.1, 0.1, 0.25, 20.0),
            _raw_factor_row("000003", 0.1, 0.1, 1.0, 0.1, 0.20, 30.0),
            _raw_factor_row("000004", 0.1, 0.1, 0.1, 1.0, 0.15, 40.0),
            _raw_factor_row("000005", 0.5, 0.5, 0.5, 0.5, 0.10, 50.0),
        ]
    )


def _raw_factor_row(
    ticker: str,
    momentum: float,
    relative_strength: float,
    quality: float,
    growth: float,
    low_volatility: float,
    liquidity: float,
) -> dict[str, object]:
    return {
        "calculation_date": pd.Timestamp("2024-06-30"),
        "available_date": pd.Timestamp("2024-06-30"),
        "ticker": ticker,
        "sector": "Test",
        "momentum_raw": momentum,
        "relative_strength_raw": relative_strength,
        "quality_raw": quality,
        "growth_raw": growth,
        "low_volatility_raw": low_volatility,
        "liquidity_raw": liquidity,
    }
