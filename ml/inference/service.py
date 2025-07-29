# ml/inference/service.py
import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ml.models.lstm_attention import LSTMWithAttention

MODEL_DIR = "ml/saved_models"
SEQ_LEN = 160


def load_model_bundle(ticker: str, interval: str):
    """
    Wczytuje model i scaler na podstawie tickera i interwału.
    """
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}.pth")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}_scaler.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Brak scalera: {scaler_path}")

    model = LSTMAttentionModel(input_size=1, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_, scaler.scale_ = np.load(scaler_path, allow_pickle=True)

    return model, scaler

def forecast_next(model, scaler, data: pd.DataFrame, column: str = "close") -> float:
    """
    Wykonuje predykcję kolejnej wartości na podstawie ostatnich danych.
    """
    sequence = data[column].values[-SEQUENCE_LENGTH:]
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
    sequence_scaled = torch.tensor(sequence_scaled.reshape(1, SEQUENCE_LENGTH, 1)).float()

    with torch.no_grad():
        prediction = model(sequence_scaled).numpy().flatten()[0]

    return scaler.inverse_transform([[prediction]])[0][0]


def _latest_raw_csv(ticker: str, interval: str) -> str:
    pattern = f"data-pipelines/feature_stores/data/raw/{ticker}_{interval}_*.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono danych raw dla wzorca: {pattern}")
    return max(files, key=os.path.getctime)


def _prepare_series_with_ma_and_scaler(df: pd.DataFrame):
    """
    Czyści dane, stosuje MA(10) na 'close', zwraca:
    - df_clean: DataFrame po MA(10) i dropna (ma kolumny: date, close_ma)
    - series_scaled: numpy array (float32) ze zeskalowaną 'close_ma'
    - scaler: fitted MinMaxScaler
    """
    d = df.copy()
    d.columns = d.columns.str.lower()
    if "close" not in d.columns:
        raise ValueError("Brak kolumny 'close' w danych.")
    if "date" not in d.columns:
        # nie przerywamy, ale stworzymy sztuczne indeksy (mniej czytelne wykresy)
        d["date"] = range(len(d))

    # MA(10) na CLOSE
    d["close_ma"] = d["close"].rolling(window=10).mean()
    d = d.dropna(subset=["close_ma"]).reset_index(drop=True)

    # Skalowanie na potrzeby modelu
    scaler = MinMaxScaler()
    d["close_scaled"] = scaler.fit_transform(d[["close_ma"]])
    series_scaled = d["close_scaled"].values.astype(np.float32)

    # Wyrównaj date do typu str (dla frontu)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values("date")
    d["date"] = d["date"].ffill()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return d[["date", "close_ma", "close_scaled"]].copy(), series_scaled, scaler


def lstm_forecast_service(
    ticker: str,
    interval: str,
    n_steps: int = 100,
    seq_len: int = SEQ_LEN,
) -> dict:
    """
    Ładuje wytrenowany model i generuje prognozę n_steps punktów do PRZODU
    na bazie ostatnich seq_len punktów.
    Zwraca prognozy w jednostkach ceny (po odwrotnym skalowaniu).
    """
    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Brak modelu: {model_path}. Najpierw wywołaj POST /ml/train."
        )

    csv_path = _latest_raw_csv(ticker, interval)
    df = pd.read_csv(csv_path, sep=";")
    df_clean, series_scaled, scaler = _prepare_series_with_ma_and_scaler(df)

    if len(series_scaled) < seq_len:
        raise ValueError(f"Za mało danych ({len(series_scaled)}) vs wymagane seq_len={seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # wejście: ostatnie seq_len
    forecast_input = torch.tensor(series_scaled[-seq_len:], dtype=torch.float32)\
                          .unsqueeze(0).unsqueeze(-1).to(device)
    forecast_scaled = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(forecast_input)            # (1, 1)
            forecast_scaled.append(pred.item())
            pred_tensor = pred.unsqueeze(1)         # (1, 1, 1)
            forecast_input = torch.cat((forecast_input[:, 1:, :], pred_tensor), dim=1)

    forecast_prices = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten().tolist()

    return {
        "ticker": ticker,
        "interval": interval,
        "n_steps": n_steps,
        "forecast": forecast_prices
    }


def lstm_backtest_service(
    ticker: str,
    interval: str,
    n_steps: int = 100,
    seq_len: int = SEQ_LEN,
) -> dict:
    """
    Szybki backtest: bierze okno *tuż przed końcem* serii (seq_len) i
    prognozuje n_steps do przodu, a potem porównuje z RZECZYWISTYMI
    danymi (ostatnie n_steps w serii). To pseudo-OOS (model był trenowany
    na pełnym zbiorze).

    Zwraca:
      - actual: rzeczywiste MA(10) ceny (ostatnie n_steps)
      - forecast: przewidywania na te same daty
      - dates: daty odpowiadające actual/forecast
      - metrics: MSE / RMSE / MAE
    """
    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Brak modelu: {model_path}. Najpierw wywołaj POST /ml/train."
        )

    csv_path = _latest_raw_csv(ticker, interval)
    df = pd.read_csv(csv_path, sep=";")
    df_clean, series_scaled, scaler = _prepare_series_with_ma_and_scaler(df)

    if len(series_scaled) < (seq_len + n_steps):
        raise ValueError(
            f"Za mało danych ({len(series_scaled)}). Wymagane: seq_len + n_steps = {seq_len + n_steps}."
        )

    # Okno wejściowe kończy się n_steps przed końcem — dzięki temu mamy "actual" w danych
    start = len(series_scaled) - (seq_len + n_steps)
    end = start + seq_len
    input_window = series_scaled[start:end]  # długość seq_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    forecast_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(x)               # (1,1)
            forecast_scaled.append(pred.item())
            x = torch.cat((x[:, 1:, :], pred.unsqueeze(1)), dim=1)

    forecast_prices = scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten()

    # Actual = ostatnie n_steps z MA(10)
    actual_prices = df_clean["close_ma"].values[-n_steps:].astype(float)
    dates = df_clean["date"].values[-n_steps:].tolist()

    # Metryki
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
