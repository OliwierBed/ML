# api/routers/ml.py
from fastapi import APIRouter, HTTPException, Query

from ml.inference.service import lstm_forecast_service, lstm_backtest_service
from ml.training.train_lstm import train_lstm_model

router = APIRouter(tags=["ml"])


@router.get("/forecast")
def forecast(
    ticker: str = Query(...),
    interval: str = Query(...),
    n_steps: int = Query(100, ge=1, le=2000),
):
    try:
        return lstm_forecast_service(ticker, interval, n_steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest")
def backtest(
    ticker: str = Query(...),
    interval: str = Query(...),
    n_steps: int = Query(100, ge=5, le=2000),
):
    try:
        return lstm_backtest_service(ticker, interval, n_steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train(
    ticker: str = Query(...),
    interval: str = Query(...),
    epochs: int = Query(25, ge=1, le=5000),
):
    try:
        train_lstm_model(ticker, interval, epochs)
        return {"status": "ok", "message": f"Wytrenowano model {ticker} {interval}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
