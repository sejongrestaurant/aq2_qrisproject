# Architecture

MUST30 is organized as a modular research and deployment stack.

## Runtime Layers

1. Data ingestion
   - `src/pipeline/collect_prices.py`
   - `src/pipeline/collect_fundamentals.py`
   - Stores cleaned price, stock, and fundamental data through SQLAlchemy repositories.

2. Research engine
   - `src/factors`: raw and composite factor scoring.
   - `src/regime`: rule-based market regime classification.
   - `src/portfolio`: stock selection, constraints, weights, and rebalance execution.
   - `src/backtest`: backtest engine, costs, benchmarks, metrics, and reports.

3. Validation
   - `src/validation/bias_checks.py`
   - `src/validation/data_quality.py`
   - `src/validation/backtest_audit.py`
   - Checks point-in-time usage, same-day execution, duplicate prices, missing costs, benchmark alignment, and other audit risks.

4. Pipeline
   - `src/pipeline/run_pipeline.py`: CLI orchestration.
   - `src/pipeline/stages.py`: stage adapters.
   - `src/pipeline/state.py`: restartable JSON state.

5. API
   - `src/api/main.py`
   - Serves universe, factors, regime, portfolio, and backtest results with Pydantic response models.

6. Dashboard
   - `dashboard/`
   - React TypeScript app using React Router, TanStack Query, and Recharts.

## Deployment

Docker Compose starts two services:

- `api`: FastAPI on port `8000`.
- `dashboard`: static React bundle served by Nginx on port `5173`.

SQLite is mounted from `./database` into the API container. Data and output folders are mounted for reproducible local operation.

## Look-Ahead Controls

- Universe rows are filtered by listing/date availability where the downstream module supports point-in-time filtering.
- Financial data uses `available_date`.
- Market regime signals are shifted so month-end data applies from the next trading day.
- Rebalance execution uses `next_open` by default.
- Backtest audit can block execution in strict mode.
