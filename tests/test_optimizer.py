"""Tests for portfolio weight optimization."""

from __future__ import annotations

import pandas as pd

from src.portfolio.optimizer import (
    WeightConstraints,
    optimize_portfolio_weights,
    validate_portfolio_weights,
)
from src.portfolio.rebalance import calculate_rebalance_weights


def test_stock_weight_cap_is_enforced() -> None:
    """A dominant score should be clipped at the stock max and redistributed."""
    portfolio = _selected_portfolio()
    portfolio.loc[0, "composite_score"] = 1_000.0

    weights = optimize_portfolio_weights(portfolio)
    stocks = weights.loc[weights["ticker"] != "CASH"]

    assert float(stocks["target_weight"].max()) <= 0.07 + 1e-12
    assert validate_portfolio_weights(weights) == []


def test_sector_weight_cap_is_enforced() -> None:
    """High-scoring names from one sector should not exceed the sector cap."""
    portfolio = _selected_portfolio()
    portfolio.loc[:14, "sector"] = "Semiconductors"
    portfolio.loc[:14, "composite_score"] = [100.0 - index for index in range(15)]

    weights = optimize_portfolio_weights(portfolio)
    sector_weights = (
        weights.loc[weights["ticker"] != "CASH"].groupby("sector")["target_weight"].sum()
    )

    assert float(sector_weights["Semiconductors"]) <= 0.25 + 1e-12
    assert validate_portfolio_weights(weights) == []


def test_kosdaq_weight_cap_is_enforced() -> None:
    """KOSDAQ aggregate weight should be capped even if KOSDAQ scores are highest."""
    portfolio = _selected_portfolio()
    portfolio.loc[:19, "market"] = "KOSDAQ"
    portfolio.loc[:19, "composite_score"] = [100.0 - index for index in range(20)]

    weights = optimize_portfolio_weights(portfolio)
    kosdaq_weight = weights.loc[weights["market"] == "KOSDAQ", "target_weight"].sum()

    assert float(kosdaq_weight) <= 0.35 + 1e-12
    assert validate_portfolio_weights(weights) == []


def test_total_weight_sums_to_one() -> None:
    """Stock weights plus cash should sum to one after iterative redistribution."""
    weights = optimize_portfolio_weights(_selected_portfolio())

    assert abs(float(weights["target_weight"].sum()) - 1.0) <= 1e-12
    assert validate_portfolio_weights(weights) == []


def test_cash_weight_uses_regime_equity_allocation() -> None:
    """Risk regime equity allocation should leave the residual in CASH."""
    weights = calculate_rebalance_weights(_selected_portfolio(), "2024-06-30", regime="Neutral")
    cash = weights.loc[weights["ticker"] == "CASH"].iloc[0]

    assert abs(float(cash["target_weight"]) - 0.20) <= 1e-12
    assert weights["rebalance_date"].eq(pd.Timestamp("2024-06-30")).all()
    assert validate_portfolio_weights(weights) == []


def test_iterative_redistribution_converges() -> None:
    """Skewed scores should converge while respecting stock, sector, and market caps."""
    portfolio = _selected_portfolio()
    portfolio.loc[:9, "composite_score"] = [500.0 - index for index in range(10)]
    portfolio.loc[:9, "sector"] = "MegaCap"
    portfolio.loc[:14, "market"] = "KOSDAQ"

    weights = optimize_portfolio_weights(
        portfolio, constraints=WeightConstraints(max_iterations=100)
    )

    assert validate_portfolio_weights(weights) == []
    assert abs(float(weights["target_weight"].sum()) - 1.0) <= 1e-12
    assert (
        "constraints_applied" in weights.loc[weights["ticker"] != "CASH", "weight_reason"].iloc[0]
    )


def test_equal_weight_and_score_weight_are_comparable() -> None:
    """Equal and score weighting should return the same schema but different stock weights."""
    portfolio = _selected_portfolio()

    equal = optimize_portfolio_weights(portfolio, method="equal_weight")
    score = optimize_portfolio_weights(portfolio, method="score_weight")

    assert equal.columns.tolist() == score.columns.tolist()
    assert not equal.loc[equal["ticker"] != "CASH", "target_weight"].equals(
        score.loc[score["ticker"] != "CASH", "target_weight"]
    )
    assert validate_portfolio_weights(equal) == []
    assert validate_portfolio_weights(score) == []


def _selected_portfolio() -> pd.DataFrame:
    tickers = [f"{idx:06d}" for idx in range(1, 31)]
    return pd.DataFrame(
        {
            "ticker": tickers,
            "composite_score": [float(31 - idx) for idx in range(1, 31)],
            "rank": list(range(1, 31)),
            "sector": [f"Sector{idx % 5}" for idx in range(30)],
            "market": ["KOSDAQ" if idx < 10 else "KOSPI" for idx in range(30)],
            "selection_reason": ["test"] * 30,
        }
    )
