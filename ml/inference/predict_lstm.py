import os, json, joblib, torch
import numpy as np
import pandas as pd
from ml.models.lstm_attn import LSTMWithAttention

def load_model_bundle(ticker, interval, models_dir="models"):
    with open(os.path.join(models_dir, f"{ticker}_{interval}_lstm_meta.json")) as f:
        meta = json.load(f)
    model = LSTMWithAttention(1, meta["hidden_dim"], meta["num_layers"])
    model.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker}_{interval}_lstm_attn.pt"), map_location="cpu"))
    model.eval()
    scaler = joblib.load(os.path.join(models_dir, f"{ticker}_{interval}_scaler.pkl"))
    return model, scaler, meta

def forecast_next(close_series, model, scaler, meta, horizon=1, device="cpu"):
    seq_len = meta["seq_len"]
    data = scaler.transform(close_series.values.reshape(-1, 1)).flatten()
    if len(data) < seq_len:
        raise ValueError("Za mało danych do sekwencji.")
    window = data[-seq_len:]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)
    # Jeżeli horizon>1, można pętlić jak w Twoim kodzie.
    inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return inv[0]
