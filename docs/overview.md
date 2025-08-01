# Application Overview

This project provides an end-to-end environment for training and evaluating trading strategies.  It consists of:

- **FastAPI backend (`api/main.py`)** – exposes REST endpoints for running ML training, forecasting, classical backtests, and for serving pre‑computed signals and equity curves.
- **Streamlit dashboard (`dashboard.py`)** – interactive frontend that consumes the API to let users trigger training, forecasts, and visualise strategy metrics.
- **Data layer** – price data is read from the configured Postgres database.  When the database is unavailable the code automatically falls back to a bundled SQLite snapshot located in `data-pipelines/feature_stores/data/database.db`.

## Running the app

```bash
uvicorn api.main:app --reload  # start backend on http://localhost:8000
streamlit run dashboard.py     # launch dashboard on http://localhost:8501
```

Using Docker Compose:
```bash
docker-compose up --build
```
The compose file starts the backend and the Streamlit frontend with the correct `BACKEND_URL`.

## Key endpoints

- `GET /tickers`, `GET /intervals`, `GET /strategies` – available options for the dashboard.
- `GET/POST /ml/train` – train LSTM model for the selected ticker and interval.
- `GET /ml/forecast` – produce price forecast using the trained LSTM model.
- `GET /ml/backtest` – evaluate the LSTM model on historical data.
- `GET /backtest/run` – run backtests for classical strategies (SMA, EMA, RSI, MACD, Bollinger).
- `GET /signals` – fetch signal series for a given strategy (computed on the fly if the DB is missing).
- `POST /signals/aggregate` – combine signals from many strategies using AND/OR/vote modes.
- `POST /equity/aggregate` – compute equity curve for aggregated signals.

## Configuration

`config/config.yaml` defines default tickers, intervals, and backtest settings.  The backend reads these values when the database is inaccessible so the application remains operational in a minimal offline mode.

## Data flow

1. **Data loading** – `db/utils/load_from_db.py` tries Postgres first and falls back to the local SQLite snapshot.  Timestamps are normalised to the `date` column for downstream processing.
2. **Signal generation** – classical strategies from `backtest/strategies/` compute signals directly from price data.  When `/signals` is requested and no signals exist in the database, they are generated on the fly.
3. **Backtesting** – `backtest/runner_db.py` wires strategies with the `BacktestEngine` to produce metrics like Sharpe, Sortino, and final equity.
4. **Frontend** – the dashboard fetches tickers/intervals/strategies at start-up and only triggers training or forecasts when the user clicks the corresponding buttons.

## Testing

Unit tests can be executed with:
```bash
pytest
```
