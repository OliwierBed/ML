from ml.inference.service import lstm_forecast_service

@router.get("/forecast")
def forecast(ticker: str, interval: str, n_steps: int = 100):
    arr = lstm_forecast_service(ticker, interval, n_steps)
    return {"ticker": ticker, "interval": interval, "forecast": arr}
