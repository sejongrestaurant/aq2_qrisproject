# MUST30 Active ETF Research Platform

MUST30 is a Korean equity strategy research project for selecting 30 stocks from a 100-stock universe, applying regime-aware equity/cash allocation, running look-ahead-safe backtests, and serving results through FastAPI and a React dashboard.

This repository is for education and research. It is not investment advice and is not an ETF prospectus or product launch package.

## Components

- Strategy engine: factor scoring, market regime classification, portfolio selection, weighting, rebalancing, and backtesting.
- Validation: data-quality and look-ahead bias checks.
- Pipeline: end-to-end orchestration with restartable stages and persisted state.
- API: FastAPI backend under `src/api`.
- Dashboard: React TypeScript frontend under `dashboard`.
- Documentation: architecture, data dictionary, factor methodology, backtest methodology, and limitations under `docs`.

## Requirements

- Python 3.12
- uv
- Bun 1.3+
- Docker Desktop, for container deployment

## Local Setup

Windows PowerShell:

```powershell
cd C:\Users\User\Documents\MUST-etf
uv sync
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Set a DART key before collecting fundamentals:

```powershell
$env:DART_API_KEY="your-dart-api-key"
```

## Quality Checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Format Python files:

```powershell
uv run ruff format .
```

## Database

Create SQLite tables:

```powershell
uv run python -m src.database.init_db
```

Default database URL:

```text
sqlite:///database/must30.db
```

## Full Pipeline

Run the integrated pipeline:

```powershell
uv run python -m src.pipeline.run_pipeline `
  --start-date 2014-01-01 `
  --end-date 2026-06-30
```

Stages:

1. `validate_universe`
2. `collect_prices`
3. `collect_fundamentals`
4. `validate_data`
5. `calculate_factors`
6. `calculate_regime`
7. `select_portfolio`
8. `calculate_weights`
9. `run_backtest`
10. `generate_report`

Run a subset:

```powershell
uv run python -m src.pipeline.run_pipeline `
  --start-date 2014-01-01 `
  --end-date 2026-06-30 `
  --from-stage calculate_factors `
  --to-stage run_backtest
```

Dry run:

```powershell
uv run python -m src.pipeline.run_pipeline `
  --start-date 2014-01-01 `
  --end-date 2026-06-30 `
  --dry-run
```

Pipeline outputs:

- State: `outputs/pipeline/pipeline_state.json`
- Config: `outputs/pipeline/pipeline_config.json`
- Logs: `outputs/pipeline/logs/pipeline.log`
- Backtest outputs: `outputs/pipeline/backtest`
- Reports: `outputs/pipeline/report`

## FastAPI Backend

```powershell
uv run uvicorn src.api.main:app --reload
```

Open:

- Health: `http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/portfolio/latest
Invoke-RestMethod http://127.0.0.1:8000/api/regime/latest
Invoke-RestMethod "http://127.0.0.1:8000/api/backtests/MUST30%20score_weight/performance"
```

## React Dashboard

```powershell
cd C:\Users\User\Documents\MUST-etf\dashboard
bun install
$env:VITE_API_URL="http://127.0.0.1:8000"
$env:VITE_USE_MOCKS="false"
bun run dev
```

Mock mode for UI development without API data:

```powershell
$env:VITE_USE_MOCKS="true"
bun run dev
```

Build:

```powershell
bun run build
```

Dashboard URL in dev mode:

```text
http://127.0.0.1:5173
```

## Docker Deployment

Build and start FastAPI, React, and the SQLite-backed app volume:

```powershell
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:5173`
- SQLite DB volume: `./database:/app/database`
- Data volume: `./data:/app/data`
- Outputs volume: `./outputs:/app/outputs`

Stop:

```powershell
docker compose down
```

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

It runs:

- `uv sync --all-groups --frozen`
- `uv run ruff check .`
- `uv run pytest`
- `uv run mypy`
- `bun install --frozen-lockfile`
- `bun run build`

## Documentation

- [Architecture](docs/architecture.md)
- [Data Dictionary](docs/data_dictionary.md)
- [Factor Methodology](docs/factor_methodology.md)
- [Backtest Methodology](docs/backtest_methodology.md)
- [Limitations](docs/limitations.md)
- [Universe Methodology](docs/universe_methodology.md)

## Data Bias Notes

- Tickers are normalized to 6-digit strings.
- Factor and backtest modules use point-in-time filters where available.
- Month-end signals are applied from the next trading day.
- Financial rows should be joined by `available_date`, not only by fiscal period.
- The present 2026-era universe can create survivorship bias in historical tests. See [Limitations](docs/limitations.md).
