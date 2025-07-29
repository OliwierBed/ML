from fastapi import APIRouter

router = APIRouter()

from ml.inference.service import lstm_forecast_service

@router.get("/forecast")
def forecast(ticker: str, interval: str, n_steps: int = 100):
    arr = lstm_forecast_service(ticker, interval, n_steps)
    return {"ticker": ticker, "interval": interval, "forecast": arr}


@router.post("/train")
def train_model(
    ticker: str = Query(...),
    interval: str = Query(...),
    epochs: int = Query(25)
):
    try:
        train_lstm_model(ticker, interval, epochs)
        return {"message": f"Model trenowany: {ticker} {interval}."}
    except Exception as e:
        return {"error": str(e)}