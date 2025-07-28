import torch
import numpy as np

def load_model_bundle(path, device):
    bundle = torch.load(path, map_location=device)
    return bundle

def forecast_next(bundle, last_seq, n_steps=100, device="cpu"):
    model_state = bundle["state_dict"]
    seq_len     = bundle["seq_len"]
    sx          = bundle["scalers"]["x"]
    sy          = bundle["scalers"]["y"]
    features    = bundle["features"]

    from ml.models.lstm_attention import LSTMWithAttention
    model = LSTMWithAttention(input_dim=len(features))
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # last_seq: numpy shape (seq_len, n_features) – już w skali SX!
    x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            y_hat = model(x).cpu().numpy().item()
            preds_scaled.append(y_hat)
            y_back = np.array(y_hat).reshape(1,1)
            y_back = np.repeat(y_back, x.shape[-1], axis=1)  # jeśli chcesz dopinać wektor featów -> najprostszy hack; docelowo dorób generator cech
            y_back = torch.tensor(y_back, dtype=torch.float32).unsqueeze(0).to(device)
            x = torch.cat([x[:,1:,:], y_back], dim=1)

    preds = sy.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return preds
