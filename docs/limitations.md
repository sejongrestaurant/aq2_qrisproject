# Limitations

This project is for research and education. It is not investment advice.

## Survivorship Bias

The current research universe is built from a 2026 기준 유니버스를 과거에 적용하는 구조를 포함할 수 있습니다. Applying a current or 2026-era universe to earlier years can overstate historical performance because companies that failed, delisted, merged, or fell out of relevance may be missing.

## Delisted Securities

상장폐지 종목 데이터에는 한계가 있습니다. If delisted stocks are absent from the source universe or price database, historical backtests can miss losses, forced exits, liquidity stress, and terminal events.

## Financial Filing Timing

재무 데이터 공시 시점 처리가 중요합니다. Financial statement data must be used only from `available_date`, not from the fiscal period end date alone. Incorrect filing-date assumptions can introduce look-ahead bias.

## Transaction Costs

거래비용과 시장충격비용은 추정치입니다. Actual ETF execution costs depend on order size, spread, liquidity, participation rate, broker, market conditions, and creation/redemption mechanics.

## Historical Performance

과거 성과가 미래 수익률을 보장하지 않습니다. Backtest results are scenario estimates based on historical assumptions and may not persist out of sample.

## Product and Regulatory Review

실제 ETF 상품 출시에는 법률, 운용, 규제 검토가 필요합니다. Index methodology, disclosure, compliance, liquidity management, tax, operational controls, and investor suitability must be reviewed by qualified professionals.

## Data Vendor Differences

Different vendors may provide different adjusted prices, corporate action treatment, listing histories, suspension flags, and financial statement revisions.

## Model Risk

Factor definitions, weighting, regime thresholds, and rebalance rules are research choices. Small changes can materially affect results, especially in concentrated or liquidity-constrained portfolios.
