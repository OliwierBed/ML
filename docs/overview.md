# Project Overview

This repository provides a small end-to-end trading research stack.  The
architecture is split into a few main pieces:

* **FastAPI backend** (`api/`)
  * exposes `/ml/train`, `/ml/forecast`, `/ml/backtest` for LSTM models
  * classical strategies and backtests are available via `/signals`,
    `/equity/aggregate` and `/backtest/run`
  * all endpoints read market data from the Postgres database configured for
    the project; no local snapshots are used at runtime
* **Backtest engine** (`backtest/`)
  * implements MACD, RSI, SMA, EMA and Bollinger strategies
  * results are computed on the fly from raw candles so no pre-computed
    signals are required
* **ML modules** (`ml/`)
  * contain the LSTM model training and inference utilities
* **Streamlit dashboard** (`dashboard.py`)
  * consumes the API and lets the user trigger training, forecasts and
    backtests interactively

## Running locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload  # backend on http://localhost:8000
streamlit run dashboard.py     # frontend on http://localhost:8501
```

The backend endpoints are documented at `http://localhost:8000/docs`.
The application requires a running Postgres instance populated with market
data. CSV files in the repository are provided only for downloading or
pre-processing data prior to inserting it into the database.  When the
backend starts it will create missing tables (such as `candles`) if they do
not yet exist, but you still need to populate them with actual market data
before training or inference endpoints can return meaningful results.
