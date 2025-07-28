# api/routers/ml.py
from fastapi import APIRouter, Query
from ml.inference.predict_lstm import predict_lstm

router = APIRouter(prefix="/ml", tags=["ml"])

@router.get("/forecast")
def lstm_forecast(
    ticker: str = Query(...),
    interval: str = Query(...),
    n_steps: int = Query(100),
    retrain: bool = Query(False)
):
    result = predict_lstm(
        ticker=ticker,
        interval=interval,
        n_steps=n_steps,
        retrain=retrain
    )
    return result
