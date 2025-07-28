import os
import json
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from ml.models.lstm_attn import LSTMWithAttention

from config.load import load_config  # jak u Ciebie

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)  # (B, T, 1)
        self.y = torch.tensor(y).unsqueeze(-1)  # (B, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_one(ticker, interval, cfg):
    processed_dir = cfg.paths.feature_stores_processed
    models_dir = getattr(cfg.paths, "models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Bierzemy **ten sam** plik co do backtestu (np. close z processed)
    # Możesz też osobny fetch zrobić.
    fname = max([f for f in os.listdir(processed_dir) if f.startswith(f"{ticker}_{interval}_") and f.endswith("_indicators.csv")])
    df = pd.read_csv(os.path.join(processed_dir, fname), sep=";")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    close = df["close"].values.reshape(-1, 1)

    seq_len = int(cfg.ml.lstm.seq_len)
    epochs = int(cfg.ml.lstm.epochs)
    batch_size = int(cfg.ml.lstm.batch_size)
    lr = float(cfg.ml.lstm.lr)
    hidden_dim = int(cfg.ml.lstm.hidden_dim)
    num_layers = int(cfg.ml.lstm.num_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close).flatten()

    # opcjonalnie smoothing (rolling mean) – zrób to konfigurowalne
    if cfg.ml.lstm.get("rolling_window", 0) > 0:
        w = cfg.ml.lstm.rolling_window
        close_scaled = pd.Series(close_scaled).rolling(w).mean().dropna().values

    X, y = create_sequences(close_scaled, seq_len)
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMWithAttention(1, hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            opt.zero_grad()
            pred = model(batch_X)
            loss = crit(pred, batch_y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"[{ticker} {interval}] epoch {ep+1}/{epochs} loss={loss_sum/len(dataloader):.6f}")

    # ZAPIS
    model_path = os.path.join(models_dir, f"{ticker}_{interval}_lstm_attn.pt")
    scaler_path = os.path.join(models_dir, f"{ticker}_{interval}_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "ticker": ticker,
        "interval": interval,
        "seq_len": seq_len,
        "input_dim": 1,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers
    }
    with open(os.path.join(models_dir, f"{ticker}_{interval}_lstm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    cfg = load_config()
    for t in cfg.data.tickers:
        for itv in cfg.data.intervals:
            train_one(t, itv, cfg)

if __name__ == "__main__":
    main()
