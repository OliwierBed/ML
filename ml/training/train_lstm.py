# ml/training/train_lstm.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "stock_market_data"  # albo data-pipelines/feature_stores/data/raw
SAVE_DIR = "ml/saved_models"
SEQ_LEN = 160
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-3

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = torch.sum(w * out, dim=1)
        return self.fc(ctx)

def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--interval", required=True)  # na razie to tylko część nazwy pliku modelu
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--data_file", type=str, default=None,
                        help="Jeśli None -> weźmiemy stock_market_data/{ticker}.csv")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_path = args.data_file or os.path.join(DATA_DIR, f"{args.ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Brak danych: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["close"]).dropna().reset_index(drop=True)
    scaler = MinMaxScaler()
    close = scaler.fit_transform(df[["close"]]).flatten()
    close = pd.Series(close).rolling(10).mean().dropna().values.astype(np.float32)

    X, y = create_sequences(close, args.seq_len)
    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y).unsqueeze(-1)

    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[{epoch+1}/{args.epochs}] loss={running/len(dl):.6f}")

    out_path = os.path.join(SAVE_DIR, f"lstm_{args.ticker}_{args.interval}.pth")
    torch.save({"state_dict": model.state_dict(),
                "scaler_min": scaler.min_[0],
                "scaler_scale": scaler.scale_[0]}, out_path)
    print(f"✔ Zapisano model: {out_path}")

def train_lstm_model(ticker: str, interval: str, epochs: int = 25):
    model_dir = "ml/saved_models"
    os.makedirs(model_dir, exist_ok=True)

    data_path = f"data-pipelines/feature_stores/data/raw/{ticker}_{interval}_*.csv"
    import glob
    files = glob.glob(data_path)
    if not files:
        raise FileNotFoundError(f"Brak danych wejściowych do treningu: {data_path}")
    file_path = max(files, key=os.path.getctime)

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["close"] = df["close"].rolling(window=10).mean().dropna()

    scaler = MinMaxScaler()
    df["close"] = scaler.fit_transform(df[["close"]])
    data = df["close"].values.astype("float32")

    # przygotuj sekwencje
    SEQ_LEN = 160
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i + SEQ_LEN])
        y.append(data[i + SEQ_LEN])
    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y).unsqueeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X.to(device))
        loss = loss_fn(out, y.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model_path = os.path.join(model_dir, f"lstm_{ticker}_{interval}.pth")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
