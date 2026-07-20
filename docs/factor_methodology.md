# Factor Methodology

MUST30 ranks stocks by a composite factor score. The score is calculated cross-sectionally for each calculation date.

## Factor Groups

- Momentum
- Relative strength
- Quality
- Growth
- Low volatility
- Liquidity

## Scoring Process

1. Normalize tickers to 6-digit strings.
2. Exclude rows whose `available_date` is after `calculation_date`.
3. Winsorize each raw factor cross-sectionally.
4. Convert each factor to a z-score.
5. Invert low-volatility direction so lower volatility receives a higher score.
6. Combine factor z-scores with configured weights.
7. Rank by `composite_score` descending and `ticker` ascending for deterministic ties.

## Missing Data

The factor configuration controls missing data handling:

- `exclude`: require all factors.
- `available_weight_rescale`: use available factors and rescale weights.
- `median_impute`: fill missing values with the date-level median.

Portfolio selection requires at least the configured minimum number of factor observations. The default selector requirement is at least 4 available factor values.

## Point-In-Time Rule

Financial and factor data must be visible only on or after `available_date`. A score calculated for a past rebalance must not use a future filing, revised future value, or future universe membership.
