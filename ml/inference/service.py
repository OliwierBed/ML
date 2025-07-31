# ml/inference/service.py
import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ml.models.lstm_attention import LSTMWithAttention
from db.utils.load_from_db import load_data_from_db

MODEL_DIR = "ml/saved_models"
SEQ_LEN = 160


def _prepare_series_with_ma_and_scaler(df: pd.DataFrame):
    d = df.copy()
    d.columns = d.columns.str.lower()
    if "close" not in d.columns:
        raise ValueError("Brak kolumny 'close' w danych.")
    if "date" not in d.columns:
        d["date"] = range(len(d))

    d["close_ma"] = d["close"].rolling(window=10).mean()
    d = d.dropna(subset=["close_ma"]).reset_index(drop=True)

    scaler = MinMaxScaler()
    d["close_scaled"] = scaler.fit_transform(d[["close_ma"]])
    series_scaled = d["close_scaled"].values.astype(np.float32)

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date")
    d["date"] = d["date"].ffill()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return d[["date", "close_ma", "close_scaled"]].copy(), series_scaled, scaler


def lstm_forecast_service(ticker: str, interval: str, n_steps: int = 100, seq_len: int = SEQ_LEN) -> dict:
    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}. Najpierw wywołaj POST /ml/train.")

    df = load_data_from_db(ticker=ticker, interval=interval, columns=["close"])
    df_clean, series_scaled, scaler = _prepare_series_with_ma_and_scaler(df)

    if len(series_scaled) < seq_len:
        raise ValueError(f"Za mało danych ({len(series_scaled)}) vs wymagane seq_len={seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    forecast_input = torch.tensor(series_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    forecast_scaled = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(forecast_input)
            forecast_scaled.append(pred.item())
            pred_tensor = pred.unsqueeze(1)
            forecast_input = torch.cat((forecast_input[:, 1:, :], pred_tensor), dim=1)

    forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten().tolist()

    return {
        "ticker": ticker,
        "interval": interval,
        "n_steps": n_steps,
        "forecast": forecast_prices
    }


def lstm_backtest_service(ticker: str, interval: str, n_steps: int = 100, seq_len: int = SEQ_LEN) -> dict:
    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}. Najpierw wywołaj POST /ml/train.")

    df = load_data_from_db(ticker=ticker, interval=interval, columns=["close"])
    df_clean, series_scaled, scaler = _prepare_series_with_ma_and_scaler(df)

    if len(series_scaled) < (seq_len + n_steps):
        raise ValueError(f"Za mało danych ({len(series_scaled)}). Wymagane: seq_len + n_steps = {seq_len + n_steps}.")

    start = len(series_scaled) - (seq_len + n_steps)
    end = start + seq_len
    input_window = series_scaled[start:end]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    forecast_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(x)
            forecast_scaled.append(pred.item())
            x = torch.cat((x[:, 1:, :], pred.unsqueeze(1)), dim=1)

    forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    actual_prices = df_clean["close_ma"].values[-n_steps:].astype(float)
    dates = df_clean["date"].values[-n_steps:].tolist()

    mse = float(mean_squared_error(actual_prices, forecast_prices))
    mae = float(mean_absolute_error(actual_prices, forecast_prices))
    rmse = float(np.sqrt(mse))

    return {
        "ticker": ticker,
        "interval": interval,
        "n_steps": n_steps,
        "dates": dates,
        "actual": actual_prices.tolist(),
        "forecast": forecast_prices.tolist(),
        "metrics": {"mse": mse, "rmse": rmse, "mae": mae},
        "note": "Backtest na MA(10) 'close'; szybka ewaluacja (model trenowany na pełnym zbiorze)."
    }

def load_model_bundle(ticker: str, interval: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}.pth")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}_scaler.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Brak scalera: {scaler_path}")

    model = LSTMWithAttention(input_size=1, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_, scaler.scale_ = np.load(scaler_path, allow_pickle=True)

    return model, scaler

def forecast_next(window: pd.Series, model, scaler, meta):
    """
    Prognozuje 1 wartość na podstawie sekwencji (używane w LSTMStrategy).
    """
    if len(window) < meta["seq_len"]:
        raise ValueError(f"Za krótka sekwencja: {len(window)} < {meta['seq_len']}")

    # oblicz MA(10)
    window_ma = window.rolling(10).mean().dropna()
    if len(window_ma) < meta["seq_len"]:
        raise ValueError(f"Za mało punktów MA(10): {len(window_ma)}")

    window_ma = window_ma[-meta["seq_len"]:]
    window_scaled = scaler.transform(window_ma.to_frame()).flatten().astype(np.float32)

    x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x).item()
    pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred_unscaled)