"""Data quality checks used by backtest audits."""

from __future__ import annotations

import pandas as pd

from src.validation.bias_checks import AuditIssue, passed_issue


def check_suspended_abnormal_returns(
    prices: pd.DataFrame,
    *,
    return_threshold: float = 0.02,
) -> AuditIssue:
    """Detect non-zero or abnormal returns during suspended trading periods."""
    check = "suspended_abnormal_returns"
    required = {"date", "ticker", "adjusted_close", "is_suspended"}
    if not required <= set(prices.columns):
        return _schema_issue(
            check, "prices requires date, ticker, adjusted_close, and is_suspended"
        )
    df = _price_frame(prices)
    df["return"] = df.groupby("ticker")["adjusted_close"].pct_change()
    bad = df.loc[
        df["is_suspended"].fillna(False).astype(bool) & (df["return"].abs() > return_threshold)
    ]
    return _issue(
        check,
        bad,
        "Suspended periods contain abnormal returns.",
        "Carry forward the last tradable price during suspensions and block trading.",
    )


def check_abnormal_price_jumps(
    prices: pd.DataFrame,
    *,
    jump_threshold: float = 0.30,
) -> AuditIssue:
    """Detect unusually large one-day adjusted price jumps."""
    check = "abnormal_price_jumps"
    required = {"date", "ticker", "adjusted_close"}
    if not required <= set(prices.columns):
        return _schema_issue(check, "prices requires date, ticker, and adjusted_close")
    df = _price_frame(prices)
    df["return"] = df.groupby("ticker")["adjusted_close"].pct_change()
    bad = df.loc[df["return"].abs() > jump_threshold]
    return _issue(
        check,
        bad,
        "Adjusted prices contain abnormal one-day jumps.",
        "Verify corporate actions, bad ticks, and adjusted price construction.",
    )


def check_duplicate_prices(prices: pd.DataFrame) -> AuditIssue:
    """Detect duplicate price rows by date and ticker."""
    check = "duplicate_price_data"
    required = {"date", "ticker"}
    if not required <= set(prices.columns):
        return _schema_issue(check, "prices requires date and ticker")
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    bad = df.loc[df.duplicated(["date", "ticker"], keep=False)]
    return _issue(
        check,
        bad,
        "Duplicate price rows exist for ticker/date pairs.",
        "Deduplicate raw prices before factor calculation or backtesting.",
    )


def check_portfolio_weight_sums(
    weights: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> AuditIssue:
    """Detect portfolio target or post-trade weights that do not sum to one by date."""
    check = "portfolio_weight_sum"
    weight_column = "target_weight" if "target_weight" in weights.columns else "post_trade_weight"
    if "date" not in weights.columns or weight_column not in weights.columns:
        return _schema_issue(check, "weights requires date plus target_weight or post_trade_weight")
    df = weights.copy()
    df["date"] = pd.to_datetime(df["date"])
    sums = df.groupby("date")[weight_column].sum()
    bad_dates = sums.loc[(sums - 1.0).abs() > tolerance].index
    bad = pd.DataFrame({"date": bad_dates})
    return _issue(
        check,
        bad,
        "Portfolio weights do not sum to 1.0.",
        "Normalize stock and CASH weights after every rebalance.",
    )


def check_transaction_costs_reflected(
    trades: pd.DataFrame,
    daily_results: pd.DataFrame | None = None,
) -> AuditIssue:
    """Detect non-zero trading with missing transaction costs."""
    check = "transaction_cost_missing"
    required = {"trade_value", "transaction_cost"}
    if not required <= set(trades.columns):
        return _schema_issue(check, "trades requires trade_value and transaction_cost")
    bad = trades.loc[
        (trades["trade_value"].abs() > 0.0) & (trades["transaction_cost"].fillna(0.0) <= 0.0)
    ]
    if (
        bad.empty
        and daily_results is not None
        and {"turnover", "transaction_cost"} <= set(daily_results.columns)
    ):
        daily_bad = daily_results.loc[
            (daily_results["turnover"].fillna(0.0) > 0.0)
            & (daily_results["transaction_cost"].fillna(0.0) <= 0.0)
        ].copy()
        if not daily_bad.empty:
            bad = daily_bad
    return _issue(
        check,
        bad,
        "Trades or rebalance days have turnover but no transaction cost.",
        "Apply one-way commission and market impact to both buys and sells.",
    )


def check_benchmark_date_alignment(
    strategy_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | pd.DataFrame,
) -> AuditIssue:
    """Detect dates present in strategy returns but absent from benchmark returns, or vice versa."""
    check = "benchmark_date_mismatch"
    strategy_dates = _date_index(strategy_returns)
    benchmark_dates = _date_index(benchmark_returns)
    missing_benchmark = strategy_dates.difference(benchmark_dates)
    missing_strategy = benchmark_dates.difference(strategy_dates)
    if len(missing_benchmark) == 0 and len(missing_strategy) == 0:
        return passed_issue(check, "Strategy and benchmark dates match.")
    affected = sorted(
        {
            pd.Timestamp(value).date().isoformat()
            for value in missing_benchmark.union(missing_strategy)
        }
    )
    return AuditIssue(
        severity="warning",
        check_name=check,
        passed=False,
        affected_dates=affected,
        affected_tickers=[],
        message="Strategy and benchmark return dates are not aligned.",
        suggested_fix="Inner-join strategy and benchmark on a common trading calendar before comparison.",
    )


def _price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    return df.sort_values(["ticker", "date"])


def _schema_issue(check_name: str, message: str) -> AuditIssue:
    return AuditIssue(
        severity="critical",
        check_name=check_name,
        passed=False,
        affected_dates=[],
        affected_tickers=[],
        message=message,
        suggested_fix="Provide the required columns before running this audit check.",
    )


def _issue(check_name: str, bad: pd.DataFrame, message: str, suggested_fix: str) -> AuditIssue:
    if bad.empty:
        return passed_issue(check_name, message.replace(" exist", " do not exist"))
    dates = []
    if "date" in bad.columns:
        dates = sorted({pd.Timestamp(value).date().isoformat() for value in bad["date"].dropna()})
    tickers = []
    if "ticker" in bad.columns:
        tickers = sorted({str(value).zfill(6) for value in bad["ticker"].dropna()})
    return AuditIssue(
        severity="warning",
        check_name=check_name,
        passed=False,
        affected_dates=dates,
        affected_tickers=tickers,
        message=message,
        suggested_fix=suggested_fix,
    )


def _date_index(data: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(data, pd.DataFrame) and "date" in data.columns:
        return pd.DatetimeIndex(pd.to_datetime(data["date"])).sort_values()
    return pd.DatetimeIndex(pd.to_datetime(data.index)).sort_values()


__all__ = [
    "check_abnormal_price_jumps",
    "check_benchmark_date_alignment",
    "check_duplicate_prices",
    "check_portfolio_weight_sums",
    "check_suspended_abnormal_returns",
    "check_transaction_costs_reflected",
]
