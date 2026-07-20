# Backtest Methodology

The backtest engine simulates a monthly MUST30 strategy with explicit separation between signal dates and execution dates.

## Flow

1. Build a point-in-time universe for the signal date.
2. Use factor scores available on or before the signal date.
3. Select the target portfolio.
4. Calculate the market regime from month-end confirmed data.
5. Calculate target weights, including a `CASH` row.
6. Execute trades on the next trading day, using `next_open` by default.
7. Apply transaction costs to buys and sells.
8. Mark positions daily using adjusted close when available.
9. Save daily results, monthly results, holdings, trades, selected factors, regime history, and metadata.

## Portfolio Construction

Default constraints:

- 30 stocks.
- Maximum stock weight: 7%.
- Minimum stock weight: 1%.
- Maximum sector weight: 25%.
- Maximum KOSDAQ weight: 35%.
- Weights plus cash must sum to 1.

Supported weighting methods:

- `equal_weight`
- `score_weight`
- `rank_weight`

## Market Regime Allocation

Default equity/cash allocation:

- Risk-On: 100% equity, 0% cash.
- Neutral: 80% equity, 20% cash.
- Risk-Off: 50% equity, 50% cash.

## Cost Model

The default one-way cost is 0.20%, split into:

- Commission: 0.15% default project assumption where configured.
- Market impact: 0.05% default project assumption where configured.

The backtest applies costs to both buys and sells.

## Bias Controls

- Signals use data available at the signal date.
- Execution occurs after the signal date.
- Financial data must be filtered by `available_date`.
- Listing dates and delisting interfaces prevent pre-listing usage where data is available.
- Suspended or unavailable tickers can be blocked from trading.

## Reported Metrics

Performance analytics include total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, monthly win rate, turnover, transaction cost, benchmark excess return, tracking error, information ratio, beta, and alpha.
