import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.title("ML Trading Bot Dashboard")

# ========== Wczytaj podstawowe dane z API ==========
tickers = requests.get(f"{API_URL}/tickers").json()["tickers"]
intervals = requests.get(f"{API_URL}/intervals").json()["intervals"]
strategies = requests.get(f"{API_URL}/strategies").json()["strategies"]

ticker = st.selectbox("Wybierz instrument (ticker):", tickers, index=0)
interval = st.selectbox("Wybierz interwał:", intervals, index=0)
selected_strategies = st.multiselect(
    "Wybierz strategię (możesz kilka):", strategies, default=[strategies[0]]
)

# ========== Tryb agregacji ==========
if len(selected_strategies) > 1:
    agg_mode = st.radio(
        "Tryb łączenia sygnałów:",
        options=["AND", "OR", "Głosowanie (wagi)"],
        index=0,
        horizontal=True
    )
else:
    agg_mode = "AND"

weights = None
if agg_mode == "Głosowanie (wagi)":
    st.write("Ustaw wagi dla każdej strategii:")
    weights = {s: st.slider(f"Waga: {s}", 0.0, 2.0, 1.0, 0.1) for s in selected_strategies}

# ========== Agregacja strategii ==========
if st.button("Pokaż wyniki"):
    # Pobierz metryki
    metrics_list = []
    for strat in selected_strategies:
        try:
            resp = requests.get(f"{API_URL}/results", params={
                "ticker": ticker,
                "interval": interval,
                "strategy": strat
            })
            if resp.ok:
                metrics_list.extend(resp.json())
        except Exception:
            pass

    if metrics_list:
        dfm = pd.DataFrame(metrics_list)
        if not dfm.empty:
            st.subheader("Metryki:")
            st.dataframe(dfm, hide_index=True)
    else:
        st.info("Brak metryk.")

    # Pobierz sygnały zagregowane
    mode = "and" if agg_mode == "AND" else "or" if agg_mode == "OR" else "vote"
    payload = {
        "ticker": ticker,
        "interval": interval,
        "strategies": selected_strategies,
        "mode": mode,
        "weights": weights if mode == "vote" else None,
    }
    sig_resp = requests.post(f"{API_URL}/signals/aggregate", json=payload)
    if not sig_resp.ok:
        st.error(f"Błąd: {sig_resp.text}")
    else:
        signals = sig_resp.json()
        if signals:
            dfsig = pd.DataFrame(signals)
            dfsig["date"] = pd.to_datetime(dfsig["date"])
            st.subheader("Sygnały inwestycyjne (agregowane):")
            st.line_chart(dfsig.set_index("date")["signal"])
        else:
            st.info("Brak sygnałów do pokazania.")

    # Pobierz equity curve
    eq_resp = requests.post(f"{API_URL}/equity/aggregate", json=payload)
    if not eq_resp.ok:
        st.error(f"Błąd equity: {eq_resp.text}")
    else:
        equity = eq_resp.json()
        if equity:
            dfeq = pd.DataFrame(equity)
            dfeq["date"] = pd.to_datetime(dfeq["date"])
            st.subheader("Krzywa wartości portfela (equity curve):")
            st.line_chart(dfeq.set_index("date")["equity"])
        else:
            st.info("Brak danych equity.")

# ========== Sekcja: Predykcja LSTM ==========
st.markdown("---")
st.subheader("📈 Predykcja LSTM")

if st.button("Wygeneruj predykcję LSTM"):
    try:
        # Na potrzeby demonstracji – lokalne wczytanie danych
        df = pd.read_csv(f"stock_market_data/{ticker}.csv", usecols=["close"])
        df = df.dropna().reset_index(drop=True)
        st.write("Dane historyczne:", df.tail())

        from sklearn.preprocessing import MinMaxScaler
        import torch
        import torch.nn as nn

        # Skalowanie
        scaler = MinMaxScaler()
        df["close"] = scaler.fit_transform(df[["close"]])
        df["close"] = df["close"].rolling(window=10).mean()
        df = df.dropna().reset_index(drop=True)
        data = df["close"].values.astype(np.float32)

        SEQ_LEN = 160

        class LSTMWithAttention(nn.Module):
            def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.attn = nn.Linear(hidden_dim, 1)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                out = self.fc(context)
                return out

        # Predykcja
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMWithAttention().to(device)
        model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))  # jeśli zapisany
        model.eval()

        forecast_input = torch.tensor(data[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        forecast = []

        with torch.no_grad():
            for _ in range(100):
                pred = model(forecast_input)
                forecast.append(pred.item())
                pred_tensor = pred.unsqueeze(1)
                forecast_input = torch.cat((forecast_input[:, 1:, :], pred_tensor), dim=1)

        forecast_rescaled = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        historical_rescaled = scaler.inverse_transform(data.reshape(-1, 1))

        st.line_chart(pd.Series(historical_rescaled.flatten(), name="Historyczne").append(
            pd.Series(forecast_rescaled.flatten(), name="Prognoza")
        ).reset_index(drop=True))
    except Exception as e:
        st.error(f"Błąd podczas predykcji: {str(e)}")
