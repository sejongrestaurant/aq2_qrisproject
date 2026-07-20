"""FastAPI application for MUST30 research data."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db_session
from src.api.schemas import (
    BacktestDailyItem,
    BacktestMonthlyItem,
    BacktestStrategyItem,
    FactorScoreItem,
    HealthResponse,
    PaginatedResponse,
    PerformanceResponse,
    PortfolioWeightItem,
    RegimeItem,
    UniverseItem,
    UniverseSummaryResponse,
)
from src.api.utils import (
    apply_date_range,
    clean_dict,
    page_meta,
    pagination_params,
    require_strategy_exists,
)
from src.backtest.metrics import (
    PerformanceConfig,
    calculate_monthly_results,
    calculate_performance_metrics,
)
from src.config.settings import get_settings
from src.database.models import BacktestDaily, FactorScore, MarketRegime, PortfolioWeight, Stock

DbSession = Annotated[Session, Depends(get_db_session)]
Pagination = Annotated[tuple[int, int], Depends(pagination_params)]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="MUST30 Active ETF API",
        description="Read-only API for universe, factors, regimes, portfolios, and backtests.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return service health and environment."""
        settings = get_settings()
        return HealthResponse(status="ok", app_name=settings.app_name, environment=settings.app_env)

    @app.get("/api/universe", response_model=list[UniverseItem])
    def get_universe(
        session: DbSession,
        pagination: Pagination,
        market: str | None = None,
        sector: str | None = None,
        is_active: bool | None = None,
    ) -> list[UniverseItem]:
        """Return paginated universe stocks."""
        limit, offset = pagination
        query = session.query(Stock).order_by(Stock.ticker)
        if market is not None:
            query = query.filter(Stock.market == market)
        if sector is not None:
            query = query.filter(Stock.sector == sector)
        if is_active is not None:
            query = query.filter(Stock.is_active == is_active)
        return list(query.offset(offset).limit(limit).all())

    @app.get("/api/universe/summary", response_model=UniverseSummaryResponse)
    def get_universe_summary(session: DbSession) -> UniverseSummaryResponse:
        """Return universe counts by market, sector, and role."""
        total = session.query(func.count(Stock.ticker)).scalar() or 0
        return UniverseSummaryResponse(
            total_count=int(total),
            market_counts=_count_by(session, Stock.market),
            sector_counts=_count_by(session, Stock.sector),
            role_counts=_count_by(session, Stock.universe_role),
        )

    @app.get("/api/factors/latest", response_model=list[FactorScoreItem])
    def get_latest_factors(session: DbSession, pagination: Pagination) -> list[FactorScoreItem]:
        """Return latest available factor scores."""
        latest = session.query(func.max(FactorScore.calculation_date)).scalar()
        if latest is None:
            return []
        return _factor_query(session, latest, pagination)

    @app.get("/api/factors/{calculation_date}", response_model=list[FactorScoreItem])
    def get_factors_by_date(
        calculation_date: date,
        session: DbSession,
        pagination: Pagination,
    ) -> list[FactorScoreItem]:
        """Return factor scores for a specific calculation date."""
        rows = _factor_query(session, calculation_date, pagination)
        if not rows:
            raise HTTPException(
                status_code=404, detail=f"factor scores not found: {calculation_date}"
            )
        return rows

    @app.get("/api/regime/latest", response_model=RegimeItem)
    def get_latest_regime(session: DbSession) -> RegimeItem:
        """Return latest market regime row."""
        row = session.query(MarketRegime).order_by(MarketRegime.date.desc()).first()
        if row is None:
            raise HTTPException(status_code=404, detail="regime history not found")
        return row

    @app.get("/api/regime/history", response_model=list[RegimeItem])
    def get_regime_history(
        session: DbSession,
        pagination: Pagination,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RegimeItem]:
        """Return market regime history with optional date range."""
        limit, offset = pagination
        query = apply_date_range(
            session.query(MarketRegime), MarketRegime.date, start_date, end_date
        )
        return list(query.order_by(MarketRegime.date).offset(offset).limit(limit).all())

    @app.get("/api/portfolio/latest", response_model=list[PortfolioWeightItem])
    def get_latest_portfolio(
        session: DbSession, pagination: Pagination
    ) -> list[PortfolioWeightItem]:
        """Return latest stored portfolio target weights."""
        latest = session.query(func.max(PortfolioWeight.rebalance_date)).scalar()
        if latest is None:
            return []
        return _portfolio_query(session, latest, pagination)

    @app.get("/api/portfolio/{rebalance_date}", response_model=list[PortfolioWeightItem])
    def get_portfolio_by_date(
        rebalance_date: date,
        session: DbSession,
        pagination: Pagination,
    ) -> list[PortfolioWeightItem]:
        """Return portfolio target weights for a rebalance date."""
        rows = _portfolio_query(session, rebalance_date, pagination)
        if not rows:
            raise HTTPException(status_code=404, detail=f"portfolio not found: {rebalance_date}")
        return rows

    @app.get("/api/backtests", response_model=list[BacktestStrategyItem])
    def get_backtests(session: DbSession, pagination: Pagination) -> list[BacktestStrategyItem]:
        """Return available stored backtest strategies."""
        limit, offset = pagination
        rows = (
            session.query(
                BacktestDaily.strategy_name,
                func.min(BacktestDaily.date).label("start_date"),
                func.max(BacktestDaily.date).label("end_date"),
                func.count(BacktestDaily.id).label("observation_count"),
            )
            .group_by(BacktestDaily.strategy_name)
            .order_by(BacktestDaily.strategy_name)
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            BacktestStrategyItem(
                strategy_name=row.strategy_name,
                start_date=row.start_date,
                end_date=row.end_date,
                observation_count=int(row.observation_count),
            )
            for row in rows
        ]

    @app.get("/api/backtests/{strategy_name}/daily", response_model=list[BacktestDailyItem])
    def get_backtest_daily(
        strategy_name: str,
        session: DbSession,
        pagination: Pagination,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[BacktestDailyItem]:
        """Return paginated daily backtest rows. Response size is capped by API settings."""
        require_strategy_exists(session, BacktestDaily, strategy_name)
        limit, offset = pagination
        query = session.query(BacktestDaily).filter(BacktestDaily.strategy_name == strategy_name)
        query = apply_date_range(query, BacktestDaily.date, start_date, end_date)
        return list(query.order_by(BacktestDaily.date).offset(offset).limit(limit).all())

    @app.get("/api/backtests/{strategy_name}/monthly", response_model=list[BacktestMonthlyItem])
    def get_backtest_monthly(
        strategy_name: str,
        session: DbSession,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[BacktestMonthlyItem]:
        """Return monthly backtest returns aggregated from daily rows."""
        daily = _daily_frame(session, strategy_name, start_date, end_date)
        monthly = calculate_monthly_results(daily)
        return [BacktestMonthlyItem(**clean_dict(row)) for row in monthly.to_dict(orient="records")]

    @app.get("/api/backtests/{strategy_name}/performance", response_model=PerformanceResponse)
    def get_backtest_performance(
        strategy_name: str,
        session: DbSession,
        start_date: date | None = None,
        end_date: date | None = None,
        risk_free_rate: float = Query(default=0.0),
    ) -> PerformanceResponse:
        """Return performance metrics calculated from stored daily return series."""
        daily = _daily_frame(session, strategy_name, start_date, end_date)
        returns = pd.Series(daily["daily_return"].to_numpy(), index=pd.to_datetime(daily["date"]))
        metrics = calculate_performance_metrics(
            returns,
            turnover=pd.Series(daily["turnover"].to_numpy(), index=returns.index),
            transaction_cost=pd.Series(daily["transaction_cost"].to_numpy(), index=returns.index),
            config=PerformanceConfig(risk_free_rate=risk_free_rate, min_observations=1),
        )
        return PerformanceResponse(
            strategy_name=strategy_name,
            metrics={key: clean_dict({"value": value})["value"] for key, value in metrics.items()},
        )

    @app.get("/api/backtests/{strategy_name}/holdings", response_model=PaginatedResponse)
    def get_backtest_holdings(
        strategy_name: str, session: DbSession, pagination: Pagination
    ) -> PaginatedResponse:
        """Return stored holdings if available. The current schema has no holdings table, so this is empty."""
        require_strategy_exists(session, BacktestDaily, strategy_name)
        limit, offset = pagination
        return PaginatedResponse(items=[], meta=page_meta(limit, offset, 0))

    @app.get("/api/backtests/{strategy_name}/trades", response_model=PaginatedResponse)
    def get_backtest_trades(
        strategy_name: str, session: DbSession, pagination: Pagination
    ) -> PaginatedResponse:
        """Return stored trades if available. The current schema has no trades table, so this is empty."""
        require_strategy_exists(session, BacktestDaily, strategy_name)
        limit, offset = pagination
        return PaginatedResponse(items=[], meta=page_meta(limit, offset, 0))


def _count_by(session: Session, column: object) -> dict[str, int]:
    """Return deterministic grouped counts."""
    rows = session.query(column, func.count()).group_by(column).order_by(column).all()
    return {str(key): int(count) for key, count in rows}


def _factor_query(
    session: Session, calculation_date: date, pagination: tuple[int, int]
) -> list[FactorScore]:
    """Return factor scores for one date."""
    limit, offset = pagination
    return list(
        session.query(FactorScore)
        .filter(FactorScore.calculation_date == calculation_date)
        .order_by(FactorScore.universe_rank, FactorScore.ticker)
        .offset(offset)
        .limit(limit)
        .all()
    )


def _portfolio_query(
    session: Session, rebalance_date: date, pagination: tuple[int, int]
) -> list[PortfolioWeight]:
    """Return portfolio weights for one date."""
    limit, offset = pagination
    return list(
        session.query(PortfolioWeight)
        .filter(PortfolioWeight.rebalance_date == rebalance_date)
        .order_by(PortfolioWeight.rank, PortfolioWeight.ticker)
        .offset(offset)
        .limit(limit)
        .all()
    )


def _daily_frame(
    session: Session,
    strategy_name: str,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Return daily rows for a strategy as a DataFrame."""
    require_strategy_exists(session, BacktestDaily, strategy_name)
    query = session.query(BacktestDaily).filter(BacktestDaily.strategy_name == strategy_name)
    query = apply_date_range(query, BacktestDaily.date, start_date, end_date)
    rows = query.order_by(BacktestDaily.date).all()
    if not rows:
        raise HTTPException(status_code=404, detail="no rows in requested date range")
    return pd.DataFrame(
        [
            {
                "date": row.date,
                "strategy_name": row.strategy_name,
                "daily_return": row.daily_return,
                "portfolio_value": row.portfolio_value,
                "benchmark_return": row.benchmark_return,
                "benchmark_value": row.benchmark_value,
                "drawdown": row.drawdown,
                "turnover": row.turnover,
                "transaction_cost": row.transaction_cost,
                "cash_weight": row.cash_weight,
            }
            for row in rows
        ]
    )


app = create_app()


__all__ = ["app", "create_app", "register_routes"]
