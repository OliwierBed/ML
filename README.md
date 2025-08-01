# ML Trading Bot

This repository contains a FastAPI backend and a Streamlit dashboard for experimenting with LSTM models and classical trading strategies.  All market data is stored in a Postgres database defined in `config/config.yaml`.

## Requirements
- Python 3.11+
- A running Postgres instance reachable with the credentials from `config/config.yaml`
- (Optional) Docker and docker compose for containerised deployment

## Running locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Postgres is running and populated with candle data (use the Alembic migrations in `alembic/` and your preferred ingestion script).
3. Start the backend:
   ```bash
   uvicorn api.main:app --reload
   ```
4. Start the dashboard:
   ```bash
   streamlit run dashboard.py
   ```
   The dashboard expects the backend on `http://localhost:8000` by default. When running inside Docker, `BACKEND_URL` is provided via `docker-compose.yml`.

## Docker compose
The repository ships with a compose file that starts Postgres, the backend and the frontend:
```bash
docker compose up --build
```

## API overview
- `POST /ml/train` or `GET /ml/train` – train LSTM model for a given ticker and interval
- `GET /ml/forecast` – forecast future prices using a trained model
- `GET /ml/backtest` – simple backtest of LSTM forecast
- `/signals`, `/signals/aggregate`, `/equity/aggregate` – classical indicator strategies

CSV files in the repository are only for downloading or preprocessing raw data. The application always reads at runtime from Postgres.

For a more detailed explanation of the architecture and workflow, see [docs/overview.md](docs/overview.md).
