# Data Dictionary

All equity tickers must be stored as 6-character strings, for example `005930`.

## Universe

| Column | Type | Description |
| --- | --- | --- |
| `rank` | integer | Universe ranking used for research universe construction. |
| `ticker` | string | 6-digit Korean stock code. |
| `company_name` | string | Korean company name. |
| `market` | string | `KOSPI` or `KOSDAQ`. |
| `sector` | string | Strategy sector bucket. |
| `industry` | string | More detailed industry label. |
| `investment_theme` | string | Research theme. |
| `universe_role` | string | `Core`, `Growth`, `Cyclical`, or `Defensive`. |
| `selection_reason` | string | Reason the security belongs to the research universe. |
| `data_start_date` | date | Earliest date from which the ticker can be used. |
| `is_active` | boolean | Current active flag in the research universe. |
| `notes` | string | Additional universe notes. |

## Daily Prices

| Column | Type | Description |
| --- | --- | --- |
| `date` | date | Trading date. |
| `ticker` | string | 6-digit stock code. |
| `open` | float | Open price. |
| `high` | float | High price. |
| `low` | float | Low price. |
| `close` | float | Close price. |
| `adjusted_close` | float | Adjusted close when available; otherwise close. |
| `volume` | float | Trading volume. |
| `trading_value` | float | Trading value. |
| `is_suspended` | boolean | Trading suspension flag when available. |

## Fundamentals

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | string | 6-digit stock code. |
| `fiscal_year` | integer | Fiscal year. |
| `report_code` | string | DART report code. |
| `report_date` | date | Fiscal period end date. |
| `available_date` | date | Date the filing became available to investors. |
| `revenue` | float | Revenue. |
| `operating_income` | float | Operating income. |
| `net_income` | float | Net income. |
| `total_assets` | float | Total assets. |
| `total_equity` | float | Total equity. |

## Factor Scores

| Column | Type | Description |
| --- | --- | --- |
| `calculation_date` | date | Date the factor score is calculated. |
| `available_date` | date | Date the source data became usable. |
| `ticker` | string | 6-digit stock code. |
| `*_raw` | float | Raw factor input values. |
| `*_score` | float | Winsorized and z-scored factor values. |
| `composite_score` | float | Weighted composite score. |
| `universe_rank` | integer | Deterministic rank by score and ticker. |

## Portfolio Weights

| Column | Type | Description |
| --- | --- | --- |
| `rebalance_date` | date | Signal date. |
| `ticker` | string | 6-digit stock code or `CASH`. |
| `target_weight` | float | Portfolio target weight. |
| `sector` | string | Sector bucket or `Cash`. |
| `market` | string | Market or `Cash`. |
| `weight_reason` | string | Explanation of the weight decision. |

## Backtest Results

| Column | Type | Description |
| --- | --- | --- |
| `date` | date | Trading date. |
| `strategy_name` | string | Strategy identifier. |
| `portfolio_value` | float | End-of-day portfolio value after costs. |
| `daily_return` | float | Daily return. |
| `drawdown` | float | Drawdown from prior peak. |
| `turnover` | float | Daily rebalance turnover. |
| `transaction_cost` | float | Costs paid on that date. |
| `cash_weight` | float | Cash allocation weight. |
