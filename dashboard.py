import os
import streamlit as st
import requests
import pandas as pd
import yaml

# Use backend service inside Docker; allow override for local runs.
# Default points to localhost so running the dashboard without Docker works
# out of the box. In docker-compose we set BACKEND_URL to the backend service.
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
with open("config/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

st.title("📊 ML Trading Bot Dashboard")


def _fetch_list(endpoint: str, key: str):
    """Helper to safely load basic lists from the backend."""
    try:
        resp = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get(key, [])
    except Exception as e:
        st.error(f"❌ Nie można połączyć się z backendem ({endpoint}): {e}")
        return []


# ========== Wczytaj podstawowe dane z API ==========
tickers = _fetch_list("tickers", "tickers") or CONFIG["data"].get("tickers", [])
intervals = _fetch_list("intervals", "intervals") or CONFIG["data"].get("intervals", [])
strategies = _fetch_list("strategies", "strategies") or list(CONFIG["backtest"].get("strategies", []))
if "ensemble" not in strategies:
    strategies.append("ensemble")

if not tickers or not intervals or not strategies:
    st.stop()

# ========== Wybór użytkownika ==========
ticker = st.selectbox("🎯 Wybierz instrument:", tickers)
interval = st.selectbox("⏱️ Wybierz interwał:", intervals)
use_all = st.checkbox("Wybierz wszystkie strategie", value=True)
selected_strategies = st.multiselect(
    "📚 Wybierz strategie:",
    strategies,
    default=strategies if use_all else [strategies[0]],
)

agg_mode = "AND"
if len(selected_strategies) > 1:
    agg_mode = st.radio("🔗 Tryb agregacji:", ["AND", "OR", "Głosowanie (wagi)"], horizontal=True)

weights = None
if agg_mode == "Głosowanie (wagi)":
    weights = {s: st.slider(f"Waga: {s}", 0.0, 2.0, 1.0, 0.1) for s in selected_strategies}

# ========== Trening LSTM ==========
st.markdown("---")
st.subheader("🧠 Trening modelu LSTM")

epochs = st.slider("Liczba epok treningowych:", 5, 200, 25, step=5)

if st.button("🔁 Wytrenuj model"):
    try:
        r = requests.post(f"{API_URL}/ml/train", json={
            "ticker": ticker,
            "interval": interval,
            "epochs": epochs,
            "n_steps": 100,
            "seq_len": 80
        })
        r.raise_for_status()
        st.success("✅ Model wytrenowany")
    except Exception as e:
        st.error(f"❌ Błąd treningu: {e}")

# ========== Predykcja ==========
st.markdown("---")
st.subheader("🔮 Predykcja LSTM")

if st.button("📈 Wygeneruj predykcję"):
    try:
        r = requests.get(f"{API_URL}/ml/forecast", params={
            "ticker": ticker, "interval": interval, "n_steps": 100
        })
        resp_json = r.json()
        if "forecast" in resp_json:
            forecast = resp_json["forecast"]
            st.line_chart(forecast)
        else:
            st.error(f"❌ Błąd predykcji: {resp_json.get('message', 'Brak danych')}")
            st.write("Odpowiedź API:", resp_json)
    except Exception as e:
        st.error(f"❌ Błąd predykcji: {e}")

# ========== Backtest predykcji ==========
st.markdown("---")
st.subheader("🧪 Backtest predykcji LSTM")

if st.button("📊 Przeprowadź backtest"):
    try:
        r = requests.get(f"{API_URL}/ml/forecast", params={
            "ticker": ticker, "interval": interval, "n_steps": 100
        })
        resp_json = r.json()
        if "forecast" in resp_json:
            forecast = pd.Series(resp_json["forecast"])
            df = pd.DataFrame({
                "forecast": forecast,
                "pct_change": forecast.pct_change().fillna(0)
            })
            df["equity"] = (1 + df["pct_change"]).cumprod()
            st.line_chart(df["equity"])
        else:
            st.error(f"❌ Błąd backtestu: {resp_json.get('message', 'Brak danych')}")
            st.write("Odpowiedź API:", resp_json)
    except Exception as e:
        st.error(f"❌ Błąd backtestu: {e}")

# ========== Strategia klasyczna i agregacja ==========
st.markdown("---")
st.subheader("📚 Klasyczne strategie inwestycyjne")

if st.button("📊 Pokaż wyniki strategii"):
    metrics = []
    for strat in selected_strategies:
        try:
            resp = requests.get(f"{API_URL}/results", params={
                "ticker": ticker,
                "interval": interval,
                "strategy": strat
            })
            if resp.ok:
                metrics.extend(resp.json())
        except:
            continue

    if metrics:
        df = pd.DataFrame(metrics)
        st.write("📈 Metryki strategii:")
        st.dataframe(df, hide_index=True)
    else:
        st.info("Brak metryk dla tych strategii.")

    # Agregacja sygnałów
    try:
        mode = "and" if agg_mode == "AND" else "or" if agg_mode == "OR" else "vote"
        payload = {
            "ticker": ticker,
            "interval": interval,
            "strategies": selected_strategies,
            "mode": mode,
            "weights": weights if mode == "vote" else None
        }
        sig_resp = requests.post(f"{API_URL}/signals/aggregate", json=payload)
        if sig_resp.ok:
            dfsig = pd.DataFrame(sig_resp.json())
            dfsig["date"] = pd.to_datetime(dfsig["date"])
            st.line_chart(dfsig.set_index("date")["signal"])
        else:
            st.warning(f"Nie udało się pobrać sygnałów: {sig_resp.text}")
    except Exception as e:
        st.error(f"Błąd agregacji sygnałów: {e}")

    # Agregacja equity curve
    try:
        eq_resp = requests.post(f"{API_URL}/equity/aggregate", json=payload)
        if eq_resp.ok:
            dfeq = pd.DataFrame(eq_resp.json())
            dfeq["date"] = pd.to_datetime(dfeq["date"])
            st.line_chart(dfeq.set_index("date")["equity"])
        else:
            st.warning(f"Nie udało się pobrać equity: {eq_resp.text}")
    except Exception as e:
        st.error(f"Błąd agregacji equity: {e}")
