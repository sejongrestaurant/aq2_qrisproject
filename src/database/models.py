"""SQLAlchemy 2.0 ORM models for the active ETF research database."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class TimestampMixin:
    """Common creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Stock(TimestampMixin, Base):
    """Stock master record for the investable universe."""

    __tablename__ = "stocks"

    ticker: Mapped[str] = mapped_column(String(6), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(120), nullable=False)
    market: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    sector: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    industry: Mapped[str] = mapped_column(String(160), nullable=False)
    investment_theme: Mapped[str] = mapped_column(String(240), nullable=False)
    universe_role: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    listing_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    daily_prices: Mapped[list[DailyPrice]] = relationship(back_populates="stock")
    fundamentals: Mapped[list[Fundamental]] = relationship(back_populates="stock")
    factor_scores: Mapped[list[FactorScore]] = relationship(back_populates="stock")
    portfolio_weights: Mapped[list[PortfolioWeight]] = relationship(back_populates="stock")


class DailyPrice(TimestampMixin, Base):
    """Daily OHLCV data.

    Prices, volume, and trading value are stored as Float because this table is expected to become
    large and is used for vectorized time-series calculations.
    """

    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_daily_prices_date_ticker"),
        Index("ix_daily_prices_ticker_date", "ticker", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    trading_value: Mapped[float] = mapped_column(Float, nullable=False)
    is_suspended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    stock: Mapped[Stock] = relationship(back_populates="daily_prices")


class Fundamental(Base):
    """Point-in-time financial statement data.

    Accounting values use Numeric to avoid avoidable binary floating point rounding in persisted
    financial statement fields.
    """

    __tablename__ = "fundamentals"
    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "report_date",
            "available_date",
            name="uq_fundamentals_ticker_report_available",
        ),
        Index("ix_fundamentals_ticker_report_date", "ticker", "report_date"),
        Index("ix_fundamentals_ticker_available_date", "ticker", "available_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False, index=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    available_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_quarter: Mapped[int] = mapped_column(Integer, nullable=False)
    revenue: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    operating_income: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    net_income: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    total_assets: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    total_equity: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    total_debt: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    operating_cash_flow: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    shares_outstanding: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))

    stock: Mapped[Stock] = relationship(back_populates="fundamentals")


class FactorScore(Base):
    """Calculated factor values and ranks.

    Raw values and scores use Float because they are derived analytics used in ranking and vector
    operations rather than audited accounting amounts.
    """

    __tablename__ = "factor_scores"
    __table_args__ = (
        UniqueConstraint("calculation_date", "ticker", name="uq_factor_scores_date_ticker"),
        Index("ix_factor_scores_ticker_date", "ticker", "calculation_date"),
        Index("ix_factor_scores_calculation_rank", "calculation_date", "universe_rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calculation_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False, index=True)
    momentum_raw: Mapped[float | None] = mapped_column(Float)
    relative_strength_raw: Mapped[float | None] = mapped_column(Float)
    quality_raw: Mapped[float | None] = mapped_column(Float)
    growth_raw: Mapped[float | None] = mapped_column(Float)
    low_volatility_raw: Mapped[float | None] = mapped_column(Float)
    liquidity_raw: Mapped[float | None] = mapped_column(Float)
    momentum_score: Mapped[float | None] = mapped_column(Float)
    relative_strength_score: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    growth_score: Mapped[float | None] = mapped_column(Float)
    low_volatility_score: Mapped[float | None] = mapped_column(Float)
    liquidity_score: Mapped[float | None] = mapped_column(Float)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    universe_rank: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    stock: Mapped[Stock] = relationship(back_populates="factor_scores")


class MarketRegime(Base):
    """Daily market regime state."""

    __tablename__ = "market_regimes"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    regime: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    kospi_close: Mapped[float] = mapped_column(Float, nullable=False)
    moving_average: Mapped[float] = mapped_column(Float, nullable=False)
    volatility: Mapped[float] = mapped_column(Float, nullable=False)
    market_breadth: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)


class PortfolioWeight(Base):
    """Target portfolio weights by rebalance date."""

    __tablename__ = "portfolio_weights"
    __table_args__ = (
        UniqueConstraint("rebalance_date", "ticker", name="uq_portfolio_weights_date_ticker"),
        Index("ix_portfolio_weights_ticker_date", "ticker", "rebalance_date"),
        Index("ix_portfolio_weights_rebalance_rank", "rebalance_date", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rebalance_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("stocks.ticker"), nullable=False, index=True)
    target_weight: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    regime: Mapped[str] = mapped_column(String(40), nullable=False)
    selection_reason: Mapped[str] = mapped_column(String(500), nullable=False)

    stock: Mapped[Stock] = relationship(back_populates="portfolio_weights")


class BacktestDaily(Base):
    """Daily backtest result series."""

    __tablename__ = "backtest_daily"
    __table_args__ = (
        Index("ix_backtest_daily_strategy_date", "strategy_name", "date"),
        UniqueConstraint("date", "strategy_name", name="uq_backtest_daily_date_strategy"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    daily_return: Mapped[float] = mapped_column(Float, nullable=False)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_return: Mapped[float | None] = mapped_column(Float)
    benchmark_value: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    turnover: Mapped[float] = mapped_column(Float, nullable=False)
    transaction_cost: Mapped[float] = mapped_column(Float, nullable=False)
    cash_weight: Mapped[float] = mapped_column(Float, nullable=False)


__all__ = [
    "BacktestDaily",
    "Base",
    "DailyPrice",
    "FactorScore",
    "Fundamental",
    "MarketRegime",
    "PortfolioWeight",
    "Stock",
]
