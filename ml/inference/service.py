# ml/inference/service.py
import os
import torch
import pandas as pd
from ml.models.lstm_attention import LSTMWithAttention
from ml.preprocessing.utils import load_lstm_data
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "ml/saved_models"
SEQ_LEN = 160

def lstm_forecast_service(ticker: str, interval: str, n_steps: int = 100) -> dict:
    model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}_{interval}.pth")
    data_path = f"data-pipelines/feature_stores/data/raw/{ticker}_{interval}_*.csv"

    # Znajdź najnowszy plik z danymi
    import glob
    files = glob.glob(data_path)
    if not files:
        raise FileNotFoundError(f"Brak danych dla {ticker} {interval}")
    file_path = max(files, key=os.path.getctime)

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["close"] = df["close"].rolling(window=10).mean().dropna()

    scaler = MinMaxScaler()
    df["close"] = scaler.fit_transform(df[["close"]])
    data = df["close"].values.astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMWithAttention().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}. Poproś ML o trening.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    forecast_input = torch.tensor(data[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    forecast = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(forecast_input)
            forecast.append(pred.item())
            forecast_input = torch.cat((forecast_input[:, 1:, :], pred.unsqueeze(1)), dim=1)

    forecast_rescaled = scaler.inverse_transform(pd.DataFrame(forecast))
    return {"forecast": forecast_rescaled.flatten().tolist()}
