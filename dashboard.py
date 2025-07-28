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
if st.button("Pokaż wyniki", key="show_results_btn"):
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

if st.button("Wygeneruj predykcję LSTM", key="lstm_btn"):
    try:
        resp = requests.get(
            f"{API_URL}/ml/forecast",
            params={"ticker": ticker, "interval": interval, "n_steps": 100}
        )
        resp.raise_for_status()
        out = resp.json()
        st.success("Gotowe! Przewidziano kolejne 100 wartości.")
        st.line_chart(out["forecast"])
    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")
