import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from ml.models.lstm_attention import LSTMWithAttention
from ml.data.loader import load_series
from ml.data.features import build_features
from ml.data.split import make_sequences, train_val_test_split
from ml.training.metrics import rmse, mae, mape, directional_accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--seq_len", type=int, default=160)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--features", nargs="+", default=["close"])  # <- możesz tu podać np. close rsi macd itp.
    return p.parse_args()

def main():
    args = parse_args()

    df = load_series(args.ticker, args.interval)
    X, y, sx, sy = build_features(df, feature_cols=args.features, target_col="close")

    Xs, ys = make_sequences(X, y, args.seq_len)
    Xtr, ytr, Xval, yval, Xte, yte = train_val_test_split(Xs, ys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(Xval), torch.tensor(yval)), batch_size=args.batch_size, shuffle=False)
    test_tensor  = (torch.tensor(Xte).to(device), torch.tensor(yte).to(device))

    model = LSTMWithAttention(input_dim=Xs.shape[-1],
                              hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers,
                              dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_path = f"ml/saved_models/lstm_{args.ticker}_{args.interval}.pth"
    os.makedirs("ml/saved_models", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # walidacja
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                val_loss += loss_fn(pred, yb).item()

        print(f"[{epoch+1}/{args.epochs}] train={train_loss/len(train_loader):.4f} val={val_loss/len(val_loader):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "scalers": {"x": sx, "y": sy},
                "seq_len": args.seq_len,
                "features": args.features
            }, best_path)
            print(f"  ↳ zapisano: {best_path}")

    # test
    bundle = torch.load(best_path, map_location=device)
    model.load_state_dict(bundle["state_dict"])
    sx, sy = bundle["scalers"]["x"], bundle["scalers"]["y"]

    model.eval()
    with torch.no_grad():
        pred = model(test_tensor[0]).squeeze().cpu().numpy()
        true = test_tensor[1].cpu().numpy()

    # odskaluj
    pred = sy.inverse_transform(pred.reshape(-1,1)).flatten()
    true = sy.inverse_transform(true.reshape(-1,1)).flatten()

    metrics = {
        "rmse": float(rmse(true, pred)),
        "mae": float(mae(true, pred)),
        "mape": float(mape(true, pred)),
        "directional_acc": float(directional_accuracy(true, pred)),
        "best_model_path": best_path,
        "ticker": args.ticker,
        "interval": args.interval
    }
    print(metrics)
    with open(best_path.replace(".pth", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
