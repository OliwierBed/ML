# ml/inference/predict_lstm.py
from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# -----------------------
# KONFIG
# -----------------------
RAW_DIR = Path("data-pipelines/feature_stores/data/raw")
MODELS_DIR = Path("models/lstm")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEQ_LEN = 160
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 25
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Pomocnicze
# -----------------------
def find_latest_raw_file(ticker: str, interval: str) -> Path:
    """
    Znajdź NAJNOWSZY plik CSV z surowymi danymi dla danego tickera i interwału.
    Przykładowy plik: AAPL_1d_20250727_222441.csv
    """
    pattern = str(RAW_DIR / f"{ticker}_{interval}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Nie znaleziono plików dla wzorca: {pattern}.\n"
            f"Upewnij się, że uruchomiłeś wcześniej pipeline pobierania danych."
        )
    latest = max(files, key=os.path.getctime)
    return Path(latest)


def create_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # (batch, seq, 1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        # (batch, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------
# Model
# -----------------------
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        out = self.fc(context)  # (batch, 1)
        return out


# -----------------------
# IO modelu + scaler
# -----------------------
def model_bundle_paths(ticker: str, interval: str, seq_len: int) -> Tuple[Path, Path]:
    """
    Zwracamy ścieżki do plików modelu i scalera.
    """
    model_path = MODELS_DIR / f"lstm_{ticker}_{interval}_seq{seq_len}.pt"
    scaler_path = MODELS_DIR / f"scaler_{ticker}_{interval}_seq{seq_len}.pkl"
    return model_path, scaler_path


def save_model_bundle(
    ticker: str,
    interval: str,
    seq_len: int,
    model: nn.Module,
    scaler: MinMaxScaler,
) -> None:
    model_path, scaler_path = model_bundle_paths(ticker, interval, seq_len)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)


def load_model_bundle(
    ticker: str,
    interval: str,
    seq_len: int,
    device: torch.device = DEVICE
) -> Tuple[Optional[LSTMWithAttention], Optional[MinMaxScaler]]:
    model_path, scaler_path = model_bundle_paths(ticker, interval, seq_len)
    if not model_path.exists() or not scaler_path.exists():
        return None, None

    model = LSTMWithAttention()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    scaler: MinMaxScaler = joblib.load(scaler_path)
    return model, scaler


# -----------------------
# Główne API
# -----------------------
def train_or_load_model(
    ticker: str,
    interval: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    retrain: bool = False
) -> Tuple[LSTMWithAttention, MinMaxScaler, np.ndarray]:
    """
    Ładuje istniejący model + scaler, lub trenuje od zera gdy brak lub 'retrain=True'.
    Zwraca:
      - model
      - scaler
      - pełny (przeskalowany) szereg cen (do prognozowania)
    """
    # 1) Spróbuj załadować gotowy model
    if not retrain:
        model_loaded, scaler_loaded = load_model_bundle(ticker, interval, seq_len, device)
        if model_loaded is not None and scaler_loaded is not None:
            # wczytaj dane, żeby mieć 'data_scaled' do predykcji
            path = find_latest_raw_file(ticker, interval)
            df = pd.read_csv(path, sep=";")
            col = "Close" if "Close" in df.columns else "close"
            data = df[col].astype(float).values.reshape(-1, 1)
            data_scaled = scaler_loaded.transform(data).flatten().astype(np.float32)
            return model_loaded, scaler_loaded, data_scaled

    # 2) Trenuj
    path = find_latest_raw_file(ticker, interval)
    df = pd.read_csv(path, sep=";")
    col = "Close" if "Close" in df.columns else "close"
    data = df[col].astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data).flatten().astype(np.float32)

    # Rolling mean, jak w Twoim pierwszym prototypie (opcjonalnie)
    # data_scaled = pd.Series(data_scaled).rolling(window=10).mean().dropna().values.astype(np.float32)

    X, y = create_sequences(data_scaled, seq_len)
    if len(X) < 2:
        raise ValueError("Za mało danych, aby utworzyć sekwencje do treningu.")

    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMWithAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[{ticker} {interval}] Epoch {epoch+1}/{epochs} | loss={np.mean(losses):.6f}")

    # Zapisz
    save_model_bundle(ticker, interval, seq_len, model, scaler)

    return model, scaler, data_scaled


def forecast_next(
    model: LSTMWithAttention,
    scaler: MinMaxScaler,
    data_scaled: np.ndarray,
    n_steps: int = 100,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device = DEVICE,
) -> List[float]:
    """
    Prognozuje n_steps naprzód (univariate).
    Zwraca listę wartości w przestrzeni zdenormalizowanej (oryginalne ceny).
    """
    model.eval()
    with torch.no_grad():
        # Start: ostatnie seq_len punktów
        window = torch.tensor(data_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        preds = []
        for _ in range(n_steps):
            pred = model(window)  # (1, 1)
            pred_val = pred.item()
            preds.append(pred_val)

            # rozsuwamy okno i dokładamy predykcję
            pred_tensor = torch.tensor([[pred_val]], dtype=torch.float32, device=device)  # (1,1)
            pred_tensor = pred_tensor.unsqueeze(-1)  # (1,1,1)
            window = torch.cat([window[:, 1:, :], pred_tensor], dim=1)

    # odwrotne skalowanie
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten().tolist()
    return preds_inv


def predict_lstm(
    ticker: str,
    interval: str,
    n_steps: int = 100,
    seq_len: int = DEFAULT_SEQ_LEN,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retrain: bool = False,
    device: torch.device = DEVICE,
) -> Dict:
    """
    Wygodna funkcja do użycia w backendzie:
    - ładuje/trenuje model dla zadanego tickera & interwału
    - zwraca prognozę (listę wartości), plus metadane
    """
    model, scaler, data_scaled = train_or_load_model(
        ticker=ticker,
        interval=interval,
        seq_len=seq_len,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        retrain=retrain
    )

    forecast = forecast_next(
        model=model,
        scaler=scaler,
        data_scaled=data_scaled,
        n_steps=n_steps,
        seq_len=seq_len,
        device=device
    )

    return {
        "ticker": ticker,
        "interval": interval,
        "seq_len": seq_len,
        "n_steps": n_steps,
        "forecast": forecast
    }
