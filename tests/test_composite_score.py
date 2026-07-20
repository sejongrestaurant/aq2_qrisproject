"""Tests for composite factor scoring."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from src.config.factor_config import DEFAULT_FACTOR_WEIGHTS, FactorConfig
from src.database.connection import (
    SessionFactory,
    create_engine_from_url,
    create_session_factory,
    session_scope,
)
from src.database.models import Base, FactorScore
from src.database.repositories import upsert_stock
from src.factors.composite_score import (
    calculate_composite_scores,
    calculate_factor_correlation,
    calculate_sector_average_scores,
    calculate_top_n_contributions,
    upsert_composite_scores,
)


def test_factor_weight_sum_validation() -> None:
    """Invalid factor weights should fail loudly."""
    weights = DEFAULT_FACTOR_WEIGHTS.copy()
    weights["momentum"] = 0.50

    with pytest.raises(ValueError, match="sum to 1.0"):
        FactorConfig(weights=weights).validate()


def test_available_weight_rescale_nan_policy() -> None:
    """Default missing policy should rescale available weights with at least four factors."""
    factors = _factor_frame()
    factors.loc[factors["ticker"] == "005930", ["growth_raw", "liquidity_raw"]] = pd.NA
    factors.loc[
        factors["ticker"] == "000660",
        ["quality_raw", "growth_raw", "liquidity_raw"],
    ] = pd.NA

    result = calculate_composite_scores(factors)

    assert "005930" in result["ticker"].tolist()
    assert "000660" not in result["ticker"].tolist()
    assert result["composite_score"].notna().all()


def test_low_volatility_direction_is_inverted() -> None:
    """Lower low_volatility_raw values should produce higher low_volatility_score."""
    result = calculate_composite_scores(_factor_frame())
    latest = result.loc[result["calculation_date"] == pd.Timestamp("2024-06-30")]
    low_risk = latest.loc[latest["ticker"] == "005930", "low_volatility_score"].iloc[0]
    high_risk = latest.loc[latest["ticker"] == "000660", "low_volatility_score"].iloc[0]

    assert low_risk > high_risk


def test_sector_neutral_scoring_differs_from_universe_scoring() -> None:
    """Sector-neutral mode should z-score within sectors instead of the whole universe."""
    factors = _factor_frame()

    universe = calculate_composite_scores(factors, FactorConfig(scoring_mode="universe"))
    sector_neutral = calculate_composite_scores(
        factors,
        FactorConfig(scoring_mode="sector_neutral"),
    )

    universe_score = universe.loc[universe["ticker"] == "005930", "momentum_score"].iloc[0]
    sector_score = sector_neutral.loc[sector_neutral["ticker"] == "005930", "momentum_score"].iloc[
        0
    ]

    assert universe_score != sector_score


def test_composite_excludes_future_available_data() -> None:
    """Rows whose available_date is after calculation_date should not affect scores."""
    factors = _factor_frame()
    baseline = calculate_composite_scores(factors)

    modified = pd.concat(
        [
            factors,
            pd.DataFrame(
                [
                    {
                        **factors.loc[factors["ticker"] == "005930"].iloc[0].to_dict(),
                        "available_date": pd.Timestamp("2024-07-15"),
                        "momentum_raw": 999.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    changed = calculate_composite_scores(modified)

    baseline_score = baseline.loc[baseline["ticker"] == "005930", "momentum_score"].iloc[0]
    changed_score = changed.loc[changed["ticker"] == "005930", "momentum_score"].iloc[0]

    assert baseline_score == changed_score


def test_composite_upsert_is_idempotent(tmp_path: Path) -> None:
    """Re-running the same date should update factor_scores without duplicate rows."""
    session_factory = _session_factory(tmp_path)
    result = calculate_composite_scores(_factor_frame())

    with session_scope(session_factory) as session:
        for ticker in result["ticker"]:
            upsert_stock(session, _stock_values(str(ticker)))
        upsert_composite_scores(session, result)
        upsert_composite_scores(session, result)

    with session_scope(session_factory) as session:
        rows = session.scalars(select(FactorScore)).all()

    assert len(rows) == len(result)


def test_composite_analysis_helpers() -> None:
    """Contribution, sector average, top-N, and correlation helpers should return data."""
    result = calculate_composite_scores(_factor_frame())

    sector_average = calculate_sector_average_scores(result)
    top_contributions = calculate_top_n_contributions(result, n=2)
    correlations = calculate_factor_correlation(result)

    assert not sector_average.empty
    assert len(top_contributions) == 2
    assert "momentum_score" in correlations.columns


def _factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _factor_row("005930", "Technology", 1.0, 0.8, 0.6, 0.5, 0.10, 100.0),
            _factor_row("000660", "Technology", 0.5, 0.6, 0.2, 0.1, 0.30, 80.0),
            _factor_row("035720", "Internet", 0.2, 0.1, 0.9, 0.7, 0.20, 60.0),
            _factor_row("051910", "Materials", -0.1, 0.2, 0.3, 0.4, 0.25, 50.0),
            _factor_row("068270", "Healthcare", 0.7, 0.5, 0.8, 0.6, 0.15, 90.0),
        ]
    )


def _factor_row(
    ticker: str,
    sector: str,
    momentum: float,
    relative_strength: float,
    quality: float,
    growth: float,
    low_volatility: float,
    liquidity: float,
) -> dict[str, object]:
    return {
        "calculation_date": pd.Timestamp("2024-06-30"),
        "available_date": pd.Timestamp("2024-06-15"),
        "ticker": ticker,
        "sector": sector,
        "momentum_raw": momentum,
        "relative_strength_raw": relative_strength,
        "quality_raw": quality,
        "growth_raw": growth,
        "low_volatility_raw": low_volatility,
        "liquidity_raw": liquidity,
    }


def _session_factory(tmp_path: Path) -> SessionFactory:
    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'composite.db'}")
    Base.metadata.create_all(engine)
    return create_session_factory(engine)


def _stock_values(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "company_name": f"Company {ticker}",
        "market": "KOSPI",
        "sector": "Technology",
        "industry": "Industry",
        "investment_theme": "Theme",
        "universe_role": "Core",
        "listing_date": date(2010, 1, 1),
        "is_active": True,
    }
