from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import os
from ml.inference.predict_lstm import load_model_bundle, forecast_next

router = APIRouter(prefix="/ml", tags=["ml"])

@router.get("/forecast")
def forecast_price(ticker: str, interval: str):
    models_dir = "models"
    try:
        model, scaler, meta = load_model_bundle(ticker, interval, models_dir)
    except FileNotFoundError:
        raise HTTPException(404, "Model nie znaleziony — wytrenuj go najpierw.")

    processed_dir = "data-pipelines/feature_stores/data/processed"
    fname = max([f for f in os.listdir(processed_dir) if f.startswith(f"{ticker}_{interval}_") and f.endswith("_indicators.csv")])
    df = pd.read_csv(os.path.join(processed_dir, fname), sep=";")
    df.columns = map(str.lower, df.columns)

    pred = forecast_next(df["close"], model, scaler, meta)
    return {"ticker": ticker, "interval": interval, "prediction": float(pred), "last_close": float(df["close"].iloc[-1])}
