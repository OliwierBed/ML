# api/routers/ml.py
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional

from ml.training.train_lstm import train_lstm_model
from ml.inference.service import lstm_forecast_service, lstm_backtest_service

router = APIRouter(prefix="/ml", tags=["ML"])


class LSTMRequest(BaseModel):
    ticker: str
    interval: str
    epochs: Optional[int] = 25
    n_steps: Optional[int] = 100
    seq_len: Optional[int] = 160


def _run_training(ticker: str, interval: str, epochs: int, seq_len: int):
    try:
        train_lstm_model(ticker=ticker, interval=interval, epochs=epochs, seq_len=seq_len)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", "message": f"Model dla {ticker} {interval} wytrenowany."}


@router.post("/train")
def train_lstm(req: LSTMRequest):
    return _run_training(
        req.ticker, req.interval, req.epochs or 25, req.seq_len or 160
    )


@router.get("/train")
def train_lstm_get(
    ticker: str = Query(...),
    interval: str = Query(...),
    epochs: int = Query(25),
    seq_len: int = Query(160),
):
    """GET variant of model training for environments where sending a JSON body
    is inconvenient (e.g., simple browser calls)."""
    return _run_training(ticker, interval, epochs, seq_len)


@router.get("/forecast")
def forecast_lstm(
    ticker: str = Query(...),
    interval: str = Query(...),
    n_steps: int = Query(100),
    seq_len: int = Query(160)
):
    try:
        result = lstm_forecast_service(ticker, interval, n_steps, seq_len)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest")
def backtest_lstm(
    ticker: str = Query(...),
    interval: str = Query(...),
    n_steps: int = Query(100),
    seq_len: int = Query(160)
):
    try:
        result = lstm_backtest_service(ticker, interval, n_steps, seq_len)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
