import os
import numpy as np
from ml.data.loader import load_series
from ml.data.features import build_features
from ml.inference.predict_lstm import load_model_bundle, forecast_next

def lstm_forecast_service(ticker: str, interval: str, n_steps: int = 100):
    model_path = f"ml/saved_models/lstm_{ticker}_{interval}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}. Poproś ML o trening.")

    bundle = load_model_bundle(model_path, device="cpu")
    features = bundle["features"]
    seq_len  = bundle["seq_len"]

    df = load_series(ticker, interval)
    X, y, sx, sy = build_features(df, feature_cols=features, target_col="close")
    Xs = []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
    last_seq = Xs[-1]  # (seq_len, n_features)

    preds = forecast_next(bundle, last_seq, n_steps=n_steps, device="cpu")
    return preds.tolist()
