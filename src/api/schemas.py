"""Pydantic response models for the MUST30 API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    environment: str


class PageMeta(BaseModel):
    """Pagination metadata."""

    limit: int
    offset: int
    count: int


class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: list[dict[str, object]]
    meta: PageMeta


class UniverseItem(BaseModel):
    """Universe stock response item."""

    ticker: str
    company_name: str
    market: str
    sector: str
    industry: str
    investment_theme: str
    universe_role: str
    listing_date: date
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class UniverseSummaryResponse(BaseModel):
    """Universe summary response."""

    total_count: int
    market_counts: dict[str, int]
    sector_counts: dict[str, int]
    role_counts: dict[str, int]


class FactorScoreItem(BaseModel):
    """Factor score response item."""

    calculation_date: date
    ticker: str
    composite_score: float
    universe_rank: int
    momentum_score: float | None = None
    relative_strength_score: float | None = None
    quality_score: float | None = None
    growth_score: float | None = None
    low_volatility_score: float | None = None
    liquidity_score: float | None = None

    model_config = ConfigDict(from_attributes=True)


class RegimeItem(BaseModel):
    """Market regime response item."""

    date: date
    regime: str
    kospi_close: float
    moving_average: float
    volatility: float
    market_breadth: float
    score: float

    model_config = ConfigDict(from_attributes=True)


class PortfolioWeightItem(BaseModel):
    """Portfolio target weight response item."""

    rebalance_date: date
    ticker: str
    target_weight: float
    rank: int
    regime: str
    selection_reason: str

    model_config = ConfigDict(from_attributes=True)


class BacktestStrategyItem(BaseModel):
    """Available backtest strategy response item."""

    strategy_name: str
    start_date: date
    end_date: date
    observation_count: int


class BacktestDailyItem(BaseModel):
    """Daily backtest response item."""

    date: date
    strategy_name: str
    daily_return: float
    portfolio_value: float
    benchmark_return: float | None = None
    benchmark_value: float | None = None
    drawdown: float
    turnover: float
    transaction_cost: float
    cash_weight: float

    model_config = ConfigDict(from_attributes=True)


class BacktestMonthlyItem(BaseModel):
    """Monthly backtest response item."""

    month: str
    month_end_date: date
    monthly_return: float
    portfolio_value: float


class PerformanceResponse(BaseModel):
    """Backtest performance response."""

    strategy_name: str
    metrics: dict[str, float | str | None]


__all__ = [
    "BacktestDailyItem",
    "BacktestMonthlyItem",
    "BacktestStrategyItem",
    "FactorScoreItem",
    "HealthResponse",
    "PageMeta",
    "PaginatedResponse",
    "PerformanceResponse",
    "PortfolioWeightItem",
    "RegimeItem",
    "UniverseItem",
    "UniverseSummaryResponse",
]
