import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.title("ML Trading Bot Dashboard")

# Pobierz tickery/interwały/strategie
tickers = requests.get(f"{API_URL}/tickers").json()["tickers"]
intervals = requests.get(f"{API_URL}/intervals").json()["intervals"]
strategies = requests.get(f"{API_URL}/strategies").json()["strategies"]

ticker = st.selectbox("Wybierz instrument (ticker):", tickers, index=0)
interval = st.selectbox("Wybierz interwał:", intervals, index=0)
selected_strategies = st.multiselect(
    "Wybierz strategię (możesz kilka):", strategies, default=[strategies[0]]
)

# Wybór trybu łączenia sygnałów
if len(selected_strategies) > 1:
    agg_mode = st.radio(
        "Tryb łączenia sygnałów:",
        options=["AND", "OR", "Głosowanie (wagi)"],
        index=0,
        horizontal=True
    )
else:
    agg_mode = "AND"

# Ustawianie wag, jeśli głosowanie
weights = None
if agg_mode == "Głosowanie (wagi)":
    st.write("Ustaw wagi dla każdej strategii:")
    weights = {}
    for s in selected_strategies:
        weights[s] = st.slider(f"Waga: {s}", 0.0, 2.0, 1.0, 0.1)

if st.button("Pokaż wyniki"):
    # Pobierz metryki (gdy jedna strategia)
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

    # Pobierz agregowane sygnały
    mode = (
        "and" if agg_mode == "AND" else
        "or" if agg_mode == "OR" else
        "vote"
    )
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

    # Pobierz equity curve (krzywa wartości portfela dla agregatu)
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
