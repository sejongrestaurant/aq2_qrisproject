"""Tests for the FastAPI backend."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.api.deps import get_db_session
from src.api.main import create_app
from src.database.connection import create_session_factory, session_scope
from src.database.models import (
    BacktestDaily,
    Base,
    FactorScore,
    MarketRegime,
    PortfolioWeight,
    Stock,
)


def test_health_endpoint() -> None:
    """Health endpoint should return app status."""
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_universe_and_summary_endpoints() -> None:
    """Universe endpoints should read stock rows from the database."""
    client = _client()

    universe = client.get("/api/universe?limit=1").json()
    summary = client.get("/api/universe/summary").json()

    assert universe[0]["ticker"] == "000001"
    assert summary["total_count"] == 2
    assert summary["market_counts"]["KOSPI"] == 2


def test_factor_portfolio_and_regime_endpoints() -> None:
    """Factor, portfolio, and regime endpoints should return dated rows."""
    client = _client()

    factors = client.get("/api/factors/latest").json()
    dated_factors = client.get("/api/factors/2024-01-31").json()
    regime = client.get("/api/regime/latest").json()
    portfolio = client.get("/api/portfolio/2024-02-01").json()

    assert factors[0]["ticker"] == "000001"
    assert dated_factors[0]["calculation_date"] == "2024-01-31"
    assert regime["regime"] == "Risk-On"
    assert portfolio[0]["ticker"] == "000001"


def test_backtest_endpoints_and_nan_free_performance() -> None:
    """Backtest endpoints should expose daily, monthly, performance, holdings, and trades."""
    client = _client()

    strategies = client.get("/api/backtests").json()
    daily = client.get(
        "/api/backtests/strategy_a/daily?start_date=2024-01-02&end_date=2024-01-03"
    ).json()
    monthly = client.get("/api/backtests/strategy_a/monthly").json()
    performance = client.get("/api/backtests/strategy_a/performance").json()
    holdings = client.get("/api/backtests/strategy_a/holdings").json()
    trades = client.get("/api/backtests/strategy_a/trades").json()

    assert strategies[0]["strategy_name"] == "strategy_a"
    assert len(daily) == 2
    assert monthly[0]["month"] == "2024-01"
    assert "total_return" in performance["metrics"]
    assert _contains_nan_literal(performance) is False
    assert holdings["items"] == []
    assert trades["items"] == []


def test_invalid_strategy_date_and_limit_errors() -> None:
    """API should return appropriate HTTP errors for bad strategy, date, and limit."""
    client = _client()

    assert client.get("/api/backtests/missing/daily").status_code == 404
    assert client.get("/api/factors/not-a-date").status_code == 422
    assert (
        client.get(
            "/api/backtests/strategy_a/daily?start_date=2024-02-01&end_date=2024-01-01"
        ).status_code
        == 422
    )
    assert client.get("/api/backtests/strategy_a/daily?limit=5000").status_code == 413


def _client() -> TestClient:
    """Create a TestClient with an isolated in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    with session_scope(session_factory) as session:
        _seed(session)

    app = create_app()

    def override_session() -> Generator[Session]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_session
    return TestClient(app)


def _seed(session: Session) -> None:
    """Seed the test database with representative rows."""
    for ticker in ("000001", "000002"):
        session.add(
            Stock(
                ticker=ticker,
                company_name=f"Company {ticker}",
                market="KOSPI",
                sector="Tech",
                industry="Software",
                investment_theme="AI",
                universe_role="Core",
                listing_date=date(2020, 1, 1),
                is_active=True,
            )
        )
    session.add(
        FactorScore(
            calculation_date=date(2024, 1, 31),
            ticker="000001",
            momentum_raw=1.0,
            relative_strength_raw=1.0,
            quality_raw=1.0,
            growth_raw=1.0,
            low_volatility_raw=1.0,
            liquidity_raw=1.0,
            momentum_score=1.0,
            relative_strength_score=1.0,
            quality_score=1.0,
            growth_score=1.0,
            low_volatility_score=1.0,
            liquidity_score=1.0,
            composite_score=1.0,
            universe_rank=1,
        )
    )
    session.add(
        MarketRegime(
            date=date(2024, 1, 31),
            regime="Risk-On",
            kospi_close=100.0,
            moving_average=95.0,
            volatility=0.1,
            market_breadth=0.6,
            score=3.0,
        )
    )
    session.add(
        PortfolioWeight(
            rebalance_date=date(2024, 2, 1),
            ticker="000001",
            target_weight=Decimal("0.50000000"),
            rank=1,
            regime="Risk-On",
            selection_reason="test",
        )
    )
    for index, daily_return in enumerate([0.0, 0.01, -0.005], start=1):
        session.add(
            BacktestDaily(
                date=date(2024, 1, index),
                strategy_name="strategy_a",
                daily_return=daily_return,
                portfolio_value=100.0 + index,
                benchmark_return=None,
                benchmark_value=None,
                drawdown=0.0,
                turnover=0.1 if index == 2 else 0.0,
                transaction_cost=0.001 if index == 2 else 0.0,
                cash_weight=0.2,
            )
        )


def _contains_nan_literal(value: object) -> bool:
    """Return whether a nested response contains a NaN float."""
    if isinstance(value, float):
        return value != value
    if isinstance(value, dict):
        return any(_contains_nan_literal(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nan_literal(item) for item in value)
    return False
