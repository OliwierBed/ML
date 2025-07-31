# api/routers/ml.py
from fastapi import APIRouter, Query
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


@router.post("/train")
def train_lstm(req: LSTMRequest):
    try:
        train_lstm_model(
            ticker=req.ticker,
            interval=req.interval,
            epochs=req.epochs or 25,
            seq_len=req.seq_len or 160
        )
        return {"status": "success", "message": f"Model dla {req.ticker} {req.interval} wytrenowany."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
        return {"status": "error", "message": str(e)}


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
        return {"status": "error", "message": str(e)}
