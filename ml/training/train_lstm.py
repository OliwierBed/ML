import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from ml.models.lstm_attention import LSTMWithAttention

MODEL_DIR = "ml/saved_models"
SEQ_LEN = 160
BATCH_SIZE = 128
LR = 1e-3


def _latest_raw_csv(ticker: str, interval: str) -> str:
    pattern = f"data-pipelines/feature_stores/data/raw/{ticker}_{interval}_*.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono danych raw dla wzorca: {pattern}")
    return max(files, key=os.path.getctime)


def _to_supervised(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_model(ticker: str, interval: str, epochs: int = 25, seq_len: int = SEQ_LEN):
    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = _latest_raw_csv(ticker, interval)

    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.lower()

    if "close" not in df.columns:
        raise ValueError(f"Brak kolumny 'close' w pliku: {csv_path}")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["close"] = df["close"].rolling(window=10).mean()
    df = df.dropna()

    scaler = MinMaxScaler()
    df["close"] = scaler.fit_transform(df[["close"]])
    series = df["close"].values.astype(np.float32)

    if len(series) <= seq_len:
        raise ValueError(f"Za mało danych ({len(series)}). Potrzeba > {seq_len}.")

    X, y = _to_supervised(series, seq_len)
    ds = SeqDataset(X, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dl:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[{ticker} {interval}] Epoch {epoch+1}/{epochs} "
              f"Loss: {epoch_loss / len(dl):.6f}")

    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Zapisano model do: {model_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--epochs", type=int, default=25)
    args = p.parse_args()

    train_lstm_model(args.ticker, args.interval, args.epochs)
