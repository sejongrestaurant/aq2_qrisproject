"""Rule-based market regime classification."""

from __future__ import annotations

import math

import pandas as pd

from src.config.regime_config import DEFAULT_REGIME_ALLOCATIONS, RegimeConfig, RegimeName


def classify_market_regime(
    market_data: pd.DataFrame,
    universe_prices: pd.DataFrame | None = None,
    *,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Classify daily market regimes from month-end confirmed signals.

    Indicators are calculated with data available on or before each date. Month-end raw regimes are
    shifted to the next trading day before becoming the applied regime, preventing same-month
    returns from using that month's closing signal.
    """
    resolved_config = config or RegimeConfig()
    resolved_config.validate()

    indicators = _prepare_indicators(market_data, universe_prices, resolved_config)
    raw = _classify_raw_regimes(indicators, resolved_config)
    confirmed = _apply_confirmation(raw, resolved_config.confirmation_days)
    smoothed = (
        _apply_hysteresis(confirmed, resolved_config)
        if resolved_config.use_hysteresis
        else confirmed
    )
    applied = _apply_next_period_shift(smoothed)
    return _add_allocations(applied, resolved_config).reset_index(drop=True)


def calculate_regime_statistics(regime_df: pd.DataFrame) -> dict[str, object]:
    """Return regime change count and average run length in trading days."""
    if regime_df.empty or "regime" not in regime_df.columns:
        return {"regime_change_count": 0, "average_duration": 0.0, "durations": {}}

    regimes = regime_df["regime"].astype(str).reset_index(drop=True)
    change_markers = regimes.ne(regimes.shift()).fillna(True)
    group_ids = change_markers.cumsum()
    durations = regimes.groupby(group_ids).agg(["first", "size"])
    non_initial_changes = max(int(change_markers.sum()) - 1, 0)
    duration_by_regime = {
        str(regime): float(group["size"].mean())
        for regime, group in durations.groupby("first", sort=True)
    }
    return {
        "regime_change_count": non_initial_changes,
        "average_duration": float(durations["size"].mean()) if not durations.empty else 0.0,
        "durations": duration_by_regime,
    }


def get_equity_cash_allocation(
    regime: RegimeName | str,
    config: RegimeConfig | None = None,
) -> tuple[float, float]:
    """Return equity and cash weights for a regime."""
    if config is None:
        if regime not in DEFAULT_REGIME_ALLOCATIONS:
            raise ValueError(f"Unknown regime: {regime}")
        return DEFAULT_REGIME_ALLOCATIONS[regime]  # type: ignore[index]

    config.validate()
    equity_weights: dict[str, float] = {
        "Risk-On": config.risk_on_equity_weight,
        "Neutral": config.neutral_equity_weight,
        "Risk-Off": config.risk_off_equity_weight,
    }
    if regime not in equity_weights:
        raise ValueError(f"Unknown regime: {regime}")
    equity = equity_weights[str(regime)]
    return equity, 1.0 - equity


def _prepare_indicators(
    market_data: pd.DataFrame,
    universe_prices: pd.DataFrame | None,
    config: RegimeConfig,
) -> pd.DataFrame:
    required_columns = {"date", "kospi_close"}
    missing = required_columns - set(market_data.columns)
    if missing:
        raise ValueError(f"Missing market data columns: {', '.join(sorted(missing))}")

    result = market_data.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    result["kospi_close"] = pd.to_numeric(result["kospi_close"], errors="coerce")

    if "kospi_ma200" not in result.columns:
        result["kospi_ma200"] = (
            result["kospi_close"]
            .rolling(config.kospi_ma_window, min_periods=config.kospi_ma_window)
            .mean()
        )
    if "kospi_momentum_60d" not in result.columns:
        result["kospi_momentum_60d"] = result["kospi_close"].pct_change(
            config.kospi_momentum_window
        )
    if "kospi_volatility_20d" not in result.columns:
        returns = result["kospi_close"].pct_change()
        result["kospi_volatility_20d"] = returns.rolling(
            config.kospi_volatility_window, min_periods=config.kospi_volatility_window
        ).std() * math.sqrt(config.annualization_days)
    if "market_breadth" not in result.columns:
        result["market_breadth"] = _calculate_market_breadth(
            universe_prices, result["date"], config
        )

    if "kosdaq_close" in result.columns:
        result["kosdaq_close"] = pd.to_numeric(result["kosdaq_close"], errors="coerce")
        result["kosdaq_momentum_60d"] = result["kosdaq_close"].pct_change(
            config.kospi_momentum_window
        )
    else:
        result["kosdaq_momentum_60d"] = pd.NA

    numeric_columns = [
        "kospi_ma200",
        "kospi_momentum_60d",
        "kospi_volatility_20d",
        "market_breadth",
        "kosdaq_momentum_60d",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _calculate_market_breadth(
    universe_prices: pd.DataFrame | None,
    dates: pd.Series,
    config: RegimeConfig,
) -> pd.Series:
    if universe_prices is None or universe_prices.empty:
        return pd.Series(pd.NA, index=dates.index, dtype="Float64")

    required_columns = {"date", "ticker", "close"}
    missing = required_columns - set(universe_prices.columns)
    if missing:
        raise ValueError(f"Missing universe price columns: {', '.join(sorted(missing))}")

    prices = universe_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.zfill(6)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.sort_values(["ticker", "date"])
    prices["ma120"] = prices.groupby("ticker")["close"].transform(
        lambda series: series.rolling(
            config.breadth_ma_window, min_periods=config.breadth_ma_window
        ).mean()
    )
    prices["above_ma120"] = prices["close"] > prices["ma120"]
    breadth = prices.groupby("date")["above_ma120"].mean()
    return pd.to_datetime(dates).map(breadth)


def _classify_raw_regimes(indicators: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    result = indicators.copy()
    required_signal_columns = ["kospi_close", "kospi_ma200", "kospi_momentum_60d", "market_breadth"]
    result["has_sufficient_data"] = result[required_signal_columns].notna().all(axis=1)
    result["kospi_above_ma200"] = result["kospi_close"] > result["kospi_ma200"]
    result["kospi_below_ma200"] = result["kospi_close"] < result["kospi_ma200"]
    result["momentum_positive"] = result["kospi_momentum_60d"] > config.risk_on_momentum_threshold
    result["momentum_negative"] = result["kospi_momentum_60d"] < config.risk_off_momentum_threshold
    result["breadth_strong"] = result["market_breadth"] > config.risk_on_breadth_threshold
    result["breadth_weak"] = result["market_breadth"] < config.risk_off_breadth_threshold
    result["kosdaq_supportive"] = result["kosdaq_momentum_60d"] > config.risk_on_momentum_threshold
    result["kosdaq_weak"] = result["kosdaq_momentum_60d"] < config.risk_off_momentum_threshold

    risk_on = result["kospi_above_ma200"] & result["momentum_positive"] & result["breadth_strong"]
    risk_off = result["kospi_below_ma200"] & result["momentum_negative"] & result["breadth_weak"]
    result["raw_regime"] = "Neutral"
    result.loc[result["has_sufficient_data"] & risk_on, "raw_regime"] = "Risk-On"
    result.loc[result["has_sufficient_data"] & risk_off, "raw_regime"] = "Risk-Off"
    result["regime_score"] = _regime_score(result)
    return result


def _regime_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype="int64")
    for column in ("kospi_above_ma200", "momentum_positive", "breadth_strong", "kosdaq_supportive"):
        score += df[column].fillna(False).astype(int)
    for column in ("kospi_below_ma200", "momentum_negative", "breadth_weak", "kosdaq_weak"):
        score -= df[column].fillna(False).astype(int)
    score.loc[~df["has_sufficient_data"]] = 0
    return score


def _apply_confirmation(df: pd.DataFrame, confirmation_days: int) -> pd.DataFrame:
    result = df.copy()
    if confirmation_days <= 1:
        result["confirmed_regime"] = result["raw_regime"]
        return result

    confirmed: list[str] = []
    raw_values = result["raw_regime"].astype(str).tolist()
    for idx, raw_regime in enumerate(raw_values):
        start = max(idx - confirmation_days + 1, 0)
        window = raw_values[start : idx + 1]
        if len(window) == confirmation_days and all(value == raw_regime for value in window):
            confirmed.append(raw_regime)
        else:
            confirmed.append("Neutral")
    result["confirmed_regime"] = confirmed
    return result


def _apply_hysteresis(df: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    result = df.copy()
    regimes: list[str] = []
    previous = "Neutral"
    for row in result.itertuples(index=False):
        candidate = str(row.confirmed_regime)
        if not bool(row.has_sufficient_data):
            previous = "Neutral"
        elif candidate != "Neutral":
            previous = candidate
        elif previous == "Risk-On" and _keep_risk_on(row, config):
            previous = "Risk-On"
        elif previous == "Risk-Off" and _keep_risk_off(row, config):
            previous = "Risk-Off"
        else:
            previous = "Neutral"
        regimes.append(previous)
    result["confirmed_regime"] = regimes
    return result


def _keep_risk_on(row: object, config: RegimeConfig) -> bool:
    return (
        bool(row.kospi_above_ma200)
        and float(row.kospi_momentum_60d)
        > config.risk_on_momentum_threshold - config.hysteresis_momentum_buffer
        and float(row.market_breadth)
        > config.risk_on_breadth_threshold - config.hysteresis_breadth_buffer
    )


def _keep_risk_off(row: object, config: RegimeConfig) -> bool:
    return (
        bool(row.kospi_below_ma200)
        and float(row.kospi_momentum_60d)
        < config.risk_off_momentum_threshold + config.hysteresis_momentum_buffer
        and float(row.market_breadth)
        < config.risk_off_breadth_threshold + config.hysteresis_breadth_buffer
    )


def _apply_next_period_shift(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    month_end_signal = result.loc[
        result["date"].dt.to_period("M").ne(result["date"].dt.to_period("M").shift(-1))
    ]
    signal_map = {
        int(row.Index): str(row.confirmed_regime)
        for row in month_end_signal[["confirmed_regime"]].itertuples()
    }

    applied_regimes: list[str] = []
    signal_dates: list[pd.Timestamp | pd.NaT] = []
    current_regime = "Neutral"
    current_signal_date: pd.Timestamp | pd.NaT = pd.NaT
    pending_regime: str | None = None
    pending_signal_date: pd.Timestamp | None = None

    for index, row in result.iterrows():
        if pending_regime is not None:
            current_regime = pending_regime
            current_signal_date = pending_signal_date if pending_signal_date is not None else pd.NaT
            pending_regime = None
            pending_signal_date = None

        applied_regimes.append(current_regime)
        signal_dates.append(current_signal_date)

        if index in signal_map:
            pending_regime = signal_map[index]
            pending_signal_date = pd.Timestamp(row["date"])

    result["regime"] = applied_regimes
    result["signal_date"] = signal_dates
    return result


def _add_allocations(df: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    result = df.copy()
    allocations = result["regime"].map(
        lambda regime: get_equity_cash_allocation(str(regime), config)
    )
    result["equity_weight"] = allocations.map(lambda value: value[0])
    result["cash_weight"] = allocations.map(lambda value: value[1])
    return result


__all__ = [
    "calculate_regime_statistics",
    "classify_market_regime",
    "get_equity_cash_allocation",
]
