"""Monthly portfolio selection engine."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from src.portfolio.constraints import PortfolioConstraints, RelaxationStep, summarize_constraints

LOGGER = logging.getLogger(__name__)

FACTOR_VALUE_COLUMNS = (
    "momentum_score",
    "relative_strength_score",
    "quality_score",
    "growth_score",
    "low_volatility_score",
    "liquidity_score",
    "momentum_raw",
    "relative_strength_raw",
    "quality_raw",
    "growth_raw",
    "low_volatility_raw",
    "liquidity_raw",
)


@dataclass(frozen=True)
class SelectionResult:
    """Selection output bundle."""

    selected_portfolio: pd.DataFrame
    excluded_stocks: pd.DataFrame
    constraint_summary: dict[str, object]


def select_monthly_portfolio(
    factor_scores: pd.DataFrame,
    universe: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    *,
    constraints: PortfolioConstraints | None = None,
    price_history: pd.DataFrame | None = None,
) -> SelectionResult:
    """Select a monthly top-30 portfolio with point-in-time constraints.

    The selector first applies non-relaxable point-in-time eligibility filters, then walks
    candidates by descending composite score and deterministic ticker order. Soft constraints can
    be relaxed in the configured order only when the target size cannot be reached.
    """
    resolved = constraints or PortfolioConstraints()
    rebalance_ts = pd.Timestamp(rebalance_date).normalize()
    candidates = _prepare_candidates(factor_scores, universe, rebalance_ts, price_history)
    hard_passed, hard_excluded = _apply_hard_filters(candidates, rebalance_ts, resolved)

    selected = pd.DataFrame()
    excluded = pd.DataFrame()
    relaxed_steps: set[RelaxationStep] = set()
    attempts = [set()]
    cumulative: set[RelaxationStep] = set()
    for step in resolved.relaxation_order:
        cumulative = {*cumulative, step}
        attempts.append(set(cumulative))

    for relaxed_steps in attempts:
        active_constraints = resolved.relaxed(relaxed_steps)
        selected, soft_excluded = _greedy_select(hard_passed, active_constraints)
        excluded = pd.concat([hard_excluded, soft_excluded], ignore_index=True)
        if len(selected) >= resolved.target_size:
            break

    active_constraints = resolved.relaxed(relaxed_steps)
    selected = _finalize_selected(selected, resolved.target_size)
    excluded = _finalize_excluded(
        excluded, selected["ticker"].tolist() if not selected.empty else []
    )
    summary = summarize_constraints(
        selected,
        excluded,
        active_constraints,
        relaxed_steps=relaxed_steps,
    )

    if len(selected) < resolved.target_size:
        LOGGER.warning(
            "Monthly selection underfilled for %s: selected=%s target=%s reasons=%s",
            rebalance_ts.date(),
            len(selected),
            resolved.target_size,
            summary["shortage_reasons"],
        )

    return SelectionResult(
        selected_portfolio=selected,
        excluded_stocks=excluded,
        constraint_summary=summary,
    )


def select_portfolio(
    factor_scores: pd.DataFrame,
    universe: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    *,
    constraints: PortfolioConstraints | None = None,
    price_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compatibility wrapper returning the three requested objects directly."""
    result = select_monthly_portfolio(
        factor_scores,
        universe,
        rebalance_date,
        constraints=constraints,
        price_history=price_history,
    )
    return result.selected_portfolio, result.excluded_stocks, result.constraint_summary


def _prepare_candidates(
    factor_scores: pd.DataFrame,
    universe: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    price_history: pd.DataFrame | None,
) -> pd.DataFrame:
    required_factor_columns = {"ticker", "composite_score"}
    missing_factor = required_factor_columns - set(factor_scores.columns)
    if missing_factor:
        raise ValueError(f"Missing factor score columns: {', '.join(sorted(missing_factor))}")

    required_universe_columns = {"ticker", "market", "sector", "universe_role"}
    missing_universe = required_universe_columns - set(universe.columns)
    if missing_universe:
        raise ValueError(f"Missing universe columns: {', '.join(sorted(missing_universe))}")

    factors = factor_scores.copy()
    factors["ticker"] = factors["ticker"].astype(str).str.zfill(6)
    if "available_date" in factors.columns:
        factors["available_date"] = pd.to_datetime(factors["available_date"])
        factors = factors.loc[factors["available_date"] <= rebalance_date]
    if "calculation_date" in factors.columns:
        factors["calculation_date"] = pd.to_datetime(factors["calculation_date"])
        factors = factors.loc[factors["calculation_date"] <= rebalance_date]
        factors = (
            factors.sort_values(["calculation_date", "ticker"])
            .groupby("ticker", as_index=False)
            .tail(1)
        )

    normalized_universe = universe.copy()
    normalized_universe["ticker"] = normalized_universe["ticker"].astype(str).str.zfill(6)
    for date_column in ("listing_date", "data_start_date", "delisting_date"):
        if date_column in normalized_universe.columns:
            normalized_universe[date_column] = pd.to_datetime(normalized_universe[date_column])

    merged = normalized_universe.merge(factors, on="ticker", how="left", suffixes=("", "_factor"))
    merged["rebalance_date"] = rebalance_date
    merged["available_factor_count"] = _available_factor_count(merged)
    merged["listing_trading_days"] = _listing_trading_days(merged, rebalance_date, price_history)
    merged["avg_trading_value_20d"] = _liquidity_value(merged, price_history, rebalance_date)
    merged["zero_volume_ratio_60d"] = _zero_volume_ratio(merged, price_history, rebalance_date)
    return merged.sort_values(
        ["composite_score", "ticker"], ascending=[False, True], na_position="last"
    )


def _apply_hard_filters(
    candidates: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    constraints: PortfolioConstraints,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclusion_reasons: list[str | None] = []
    for row in candidates.itertuples(index=False):
        exclusion_reasons.append(_hard_exclusion_reason(row, rebalance_date, constraints))

    evaluated = candidates.copy()
    evaluated["exclusion_reason"] = exclusion_reasons
    excluded = evaluated.loc[evaluated["exclusion_reason"].notna()].copy()
    passed = (
        evaluated.loc[evaluated["exclusion_reason"].isna()]
        .drop(columns=["exclusion_reason"])
        .copy()
    )
    return passed, excluded


def _hard_exclusion_reason(
    row: object,
    rebalance_date: pd.Timestamp,
    constraints: PortfolioConstraints,
) -> str | None:
    if hasattr(row, "is_active") and not bool(row.is_active):
        return "inactive_universe_member"

    listing_date = _row_date(row, "listing_date") or _row_date(row, "data_start_date")
    if listing_date is not None and listing_date > rebalance_date:
        return "not_listed_on_rebalance_date"

    delisting_date = _row_date(row, "delisting_date")
    if delisting_date is not None and delisting_date <= rebalance_date:
        return "delisted_before_rebalance_date"

    if pd.isna(getattr(row, "composite_score", pd.NA)):
        return "missing_composite_score"

    if int(getattr(row, "available_factor_count", 0)) < constraints.min_available_factors:
        return "insufficient_factor_data"

    if int(getattr(row, "listing_trading_days", 0)) < constraints.min_listing_trading_days:
        return "insufficient_listing_history"

    avg_value = float(getattr(row, "avg_trading_value_20d", 0.0))
    if avg_value < constraints.min_avg_trading_value_20d:
        return "insufficient_trading_value"

    zero_ratio = float(getattr(row, "zero_volume_ratio_60d", 1.0))
    if zero_ratio > constraints.max_zero_volume_ratio_60d:
        return "too_many_zero_volume_days"

    return None


def _greedy_select(
    candidates: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.Series] = []
    excluded_rows: list[pd.Series] = []
    sector_counts: Counter[str] = Counter()
    market_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    selected_tickers: set[str] = set()

    ordered = candidates.sort_values(
        ["composite_score", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    for idx, row in ordered.iterrows():
        if len(selected_rows) >= constraints.target_size:
            excluded_rows.append(_exclude(row, "below_final_score_cutoff"))
            continue

        reason = _soft_exclusion_reason(
            row,
            ordered.iloc[idx + 1 :],
            selected_tickers,
            sector_counts,
            market_counts,
            role_counts,
            constraints,
        )
        if reason is not None:
            excluded_rows.append(_exclude(row, reason))
            continue

        selected_rows.append(_select(row, len(selected_rows) + 1))
        selected_tickers.add(str(row["ticker"]))
        sector_counts[str(row["sector"])] += 1
        market_counts[str(row["market"])] += 1
        role_counts[str(row["universe_role"])] += 1

    return pd.DataFrame(selected_rows), pd.DataFrame(excluded_rows)


def _soft_exclusion_reason(
    row: pd.Series,
    remaining: pd.DataFrame,
    selected_tickers: set[str],
    sector_counts: Counter[str],
    market_counts: Counter[str],
    role_counts: Counter[str],
    constraints: PortfolioConstraints,
) -> str | None:
    sector = str(row["sector"])
    market = str(row["market"])
    role = str(row["universe_role"])

    if sector_counts[sector] >= constraints.max_sector_count:
        return "sector_limit"
    if market == "KOSDAQ" and market_counts["KOSDAQ"] >= constraints.max_kosdaq_count:
        return "kosdaq_limit"

    projected_role_counts = Counter(role_counts)
    projected_role_counts[role] += 1
    projected_selected = len(selected_tickers) + 1
    slots_after = constraints.target_size - projected_selected
    if not _can_satisfy_min_roles(
        remaining,
        selected_tickers | {str(row["ticker"])},
        sector_counts + Counter({sector: 1}),
        market_counts + Counter({market: 1}),
        projected_role_counts,
        slots_after,
        constraints,
    ):
        return "reserved_for_min_role"

    return None


def _can_satisfy_min_roles(
    remaining: pd.DataFrame,
    selected_tickers: set[str],
    sector_counts: Counter[str],
    market_counts: Counter[str],
    role_counts: Counter[str],
    slots_after: int,
    constraints: PortfolioConstraints,
) -> bool:
    core_needed = max(constraints.min_core_count - role_counts["Core"], 0)
    defensive_needed = max(constraints.min_defensive_count - role_counts["Defensive"], 0)
    if core_needed + defensive_needed > slots_after:
        return False

    future_core = _future_role_capacity(
        remaining,
        "Core",
        selected_tickers,
        sector_counts,
        market_counts,
        constraints,
    )
    future_defensive = _future_role_capacity(
        remaining,
        "Defensive",
        selected_tickers,
        sector_counts,
        market_counts,
        constraints,
    )
    return future_core >= core_needed and future_defensive >= defensive_needed


def _future_role_capacity(
    remaining: pd.DataFrame,
    role: str,
    selected_tickers: set[str],
    sector_counts: Counter[str],
    market_counts: Counter[str],
    constraints: PortfolioConstraints,
) -> int:
    capacity = 0
    trial_sector_counts = Counter(sector_counts)
    trial_market_counts = Counter(market_counts)
    for row in remaining.itertuples(index=False):
        ticker = str(row.ticker)
        if ticker in selected_tickers or str(row.universe_role) != role:
            continue
        sector = str(row.sector)
        market = str(row.market)
        if trial_sector_counts[sector] >= constraints.max_sector_count:
            continue
        if market == "KOSDAQ" and trial_market_counts["KOSDAQ"] >= constraints.max_kosdaq_count:
            continue
        capacity += 1
        trial_sector_counts[sector] += 1
        trial_market_counts[market] += 1
    return capacity


def _select(row: pd.Series, rank: int) -> pd.Series:
    selected = row.copy()
    selected["rank"] = rank
    selected["selection_reason"] = (
        f"selected_rank_{rank}: composite_score={float(row['composite_score']):.6f}; "
        f"role={row['universe_role']}; sector={row['sector']}; market={row['market']}"
    )
    return selected


def _exclude(row: pd.Series, reason: str) -> pd.Series:
    excluded = row.copy()
    excluded["exclusion_reason"] = reason
    return excluded


def _finalize_selected(selected: pd.DataFrame, target_size: int) -> pd.DataFrame:
    if selected.empty:
        return selected
    selected = selected.sort_values("rank").head(target_size).reset_index(drop=True)
    return selected


def _finalize_excluded(excluded: pd.DataFrame, selected_tickers: Iterable[str]) -> pd.DataFrame:
    if excluded.empty:
        return excluded
    selected_set = set(selected_tickers)
    result = excluded.loc[~excluded["ticker"].astype(str).isin(selected_set)].copy()
    return result.sort_values(
        ["exclusion_reason", "composite_score", "ticker"],
        ascending=[True, False, True],
    ).reset_index(
        drop=True,
    )


def _available_factor_count(df: pd.DataFrame) -> pd.Series:
    columns = [column for column in FACTOR_VALUE_COLUMNS if column in df.columns]
    if not columns:
        return pd.Series(0, index=df.index)
    score_columns = [column for column in columns if column.endswith("_score")]
    preferred = score_columns or columns
    return df[preferred].notna().sum(axis=1)


def _listing_trading_days(
    candidates: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    price_history: pd.DataFrame | None,
) -> pd.Series:
    if price_history is not None and not price_history.empty:
        prices = _normalize_price_history(price_history)
        prices = prices.loc[prices["date"] <= rebalance_date]
        counts = prices.groupby("ticker")["date"].nunique()
        return candidates["ticker"].map(counts).fillna(0).astype(int)

    date_column = "listing_date" if "listing_date" in candidates.columns else "data_start_date"
    if date_column not in candidates.columns:
        return pd.Series(0, index=candidates.index)
    return candidates[date_column].map(
        lambda value: (
            len(pd.bdate_range(pd.Timestamp(value), rebalance_date)) if pd.notna(value) else 0
        )
    )


def _liquidity_value(
    candidates: pd.DataFrame,
    price_history: pd.DataFrame | None,
    rebalance_date: pd.Timestamp,
) -> pd.Series:
    for column in ("avg_trading_value_20d", "trading_value_20d_avg"):
        if column in candidates.columns:
            return pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)

    if price_history is None or price_history.empty:
        return pd.Series(0.0, index=candidates.index)

    prices = _normalize_price_history(price_history)
    last_20 = (
        prices.loc[prices["date"] <= rebalance_date]
        .sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(20)
    )
    means = last_20.groupby("ticker")["trading_value"].mean()
    return candidates["ticker"].map(means).fillna(0.0).astype(float)


def _zero_volume_ratio(
    candidates: pd.DataFrame,
    price_history: pd.DataFrame | None,
    rebalance_date: pd.Timestamp,
) -> pd.Series:
    for column in ("zero_volume_ratio_60d", "zero_volume_60d_ratio"):
        if column in candidates.columns:
            return pd.to_numeric(candidates[column], errors="coerce").fillna(1.0)

    if "zero_volume_days_60d" in candidates.columns:
        return (pd.to_numeric(candidates["zero_volume_days_60d"], errors="coerce") / 60.0).fillna(
            1.0
        )

    if price_history is None or price_history.empty:
        return pd.Series(0.0, index=candidates.index)

    prices = _normalize_price_history(price_history)
    last_60 = (
        prices.loc[prices["date"] <= rebalance_date]
        .sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(60)
    )
    ratios = last_60.groupby("ticker")["volume"].apply(lambda series: float((series == 0).mean()))
    return candidates["ticker"].map(ratios).fillna(1.0).astype(float)


def _normalize_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker"}
    missing = required - set(price_history.columns)
    if missing:
        raise ValueError(f"Missing price history columns: {', '.join(sorted(missing))}")
    prices = price_history.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"])
    if "trading_value" not in prices.columns:
        prices["trading_value"] = 0.0
    if "volume" not in prices.columns:
        prices["volume"] = 0.0
    return prices


def _row_date(row: object, column: str) -> pd.Timestamp | None:
    if not hasattr(row, column):
        return None
    value = getattr(row, column)
    if pd.isna(value):
        return None
    return pd.Timestamp(value)


__all__ = [
    "FACTOR_VALUE_COLUMNS",
    "SelectionResult",
    "select_monthly_portfolio",
    "select_portfolio",
]
