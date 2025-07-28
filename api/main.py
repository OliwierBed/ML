from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from functools import reduce
from api.routers import ml

app = FastAPI(title="TradingBot API")

app.include_router(ml.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = "data-pipelines/feature_stores/data/results"
ENSEMBLE_DIR = "data-pipelines/feature_stores/data/processed/ensemble"
PROCESSED_DIR = "data-pipelines/feature_stores/data/processed"

@app.get("/tickers")
def get_tickers():
    try:
        results = pd.read_csv(os.path.join(RESULTS_DIR, "batch_results.csv"), sep=";")
        tickers = sorted(results["ticker"].unique().tolist())
    except Exception:
        tickers = ["AAPL", "MSFT", "TSLA"]
    return {"tickers": tickers}

@app.get("/intervals")
def get_intervals():
    try:
        results = pd.read_csv(os.path.join(RESULTS_DIR, "batch_results.csv"), sep=";")
        intervals = sorted(results["interval"].unique().tolist())
    except Exception:
        intervals = ["1h", "1d", "1wk"]
    return {"intervals": intervals}

@app.get("/strategies")
def get_strategies():
    try:
        results = pd.read_csv(os.path.join(RESULTS_DIR, "batch_results.csv"), sep=";")
        strategies = sorted(list(set(results["strategy"].unique().tolist() + ["ensemble"])))
    except Exception:
        strategies = ["macd", "rsi", "sma", "ema", "bollinger", "ensemble"]
    return {"strategies": strategies}

@app.get("/results")
def get_results(
    ticker: str = Query(None),
    interval: str = Query(None),
    strategy: str = Query(None)
):
    df = pd.read_csv(os.path.join(RESULTS_DIR, "batch_results.csv"), sep=";")
    if ticker:
        df = df[df["ticker"] == ticker]
    if interval:
        df = df[df["interval"] == interval]
    if strategy and strategy != "ensemble":
        df = df[df["strategy"] == strategy]
    if strategy == "ensemble":
        try:
            ens = pd.read_csv(os.path.join(RESULTS_DIR, "ensemble_batch_results.csv"), sep=";")
            if ticker:
                ens = ens[ens["ticker"] == ticker]
            if interval:
                ens = ens[ens["interval"] == interval]
            return ens.to_dict(orient="records")
        except Exception:
            raise HTTPException(status_code=404, detail="Brak wyników ensemble.")
    return df.to_dict(orient="records")

@app.get("/signals")
def get_signals(
    ticker: str,
    interval: str,
    strategy: str = "ensemble"
):
    if strategy == "ensemble":
        fname = f"{ticker}_{interval}_ensemble_full.csv"
        path = os.path.join(ENSEMBLE_DIR, fname)
    else:
        fname = f"{ticker}_{interval}_20250724_111413_indicators.csv"
        path = os.path.join(PROCESSED_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Brak pliku sygnałów: {path}")
    df = pd.read_csv(path, sep=";")
    columns = [c for c in df.columns if c in ("date", "signal", "close")]
    return df[columns].to_dict(orient="records")

# --- NOWOŚĆ: endpoint do agregacji sygnałów ---
@app.post("/signals/aggregate")
def aggregate_signals(body: dict = Body(...)):
    ticker = body.get("ticker")
    interval = body.get("interval")
    strategies = body.get("strategies")
    mode = body.get("mode", "and")
    weights = body.get("weights", None)
    if not (ticker and interval and strategies and mode):
        raise HTTPException(status_code=400, detail="Brak wymaganych parametrów.")

    all_signals = []
    for strat in strategies:
        if strat == "ensemble":
            fname = f"{ticker}_{interval}_ensemble_full.csv"
            path = os.path.join(ENSEMBLE_DIR, fname)
        else:
            fname = f"{ticker}_{interval}_20250724_111413_indicators.csv"
            path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=";")
        if "date" not in df.columns or "signal" not in df.columns:
            continue
        df = df[["date", "signal"]].copy()
        df.rename(columns={"signal": f"signal_{strat}"}, inplace=True)
        all_signals.append(df)
    if not all_signals:
        return []
    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), all_signals)
    merged = merged.sort_values("date")
    # --- Agregacja ---
    sigcols = [c for c in merged.columns if c.startswith("signal_")]
    if mode == "and":
        merged["signal"] = merged[sigcols].apply(
            lambda row: 1 if all(x == 1 for x in row) else -1 if all(x == -1 for x in row) else 0, axis=1)
    elif mode == "or":
        merged["signal"] = merged[sigcols].apply(
            lambda row: 1 if any(x == 1 for x in row) else -1 if any(x == -1 for x in row) else 0, axis=1)
    elif mode == "vote":
        weight_vals = [weights.get(c.replace("signal_", ""), 1.0) if weights else 1.0 for c in sigcols]
        def weighted_vote(row):
            s = sum(w * (row[col] if pd.notnull(row[col]) else 0) for w, col in zip(weight_vals, sigcols))
            if s > 0: return 1
            if s < 0: return -1
            return 0
        merged["signal"] = merged.apply(weighted_vote, axis=1)
    else:
        raise HTTPException(status_code=400, detail="Nieznany tryb agregacji.")
    return merged[["date", "signal"]].to_dict(orient="records")

# --- NOWOŚĆ: endpoint do equity curve dla agregatu ---
@app.post("/equity/aggregate")
def aggregate_equity(body: dict = Body(...)):
    ticker = body.get("ticker")
    interval = body.get("interval")
    strategies = body.get("strategies")
    mode = body.get("mode", "and")
    weights = body.get("weights", None)
    initial_cash = 100000  # albo z configu

    # Wczytaj sygnały zagregowane
    all_signals = []
    closes = None
    for strat in strategies:
        if strat == "ensemble":
            fname = f"{ticker}_{interval}_ensemble_full.csv"
            path = os.path.join(ENSEMBLE_DIR, fname)
        else:
            fname = f"{ticker}_{interval}_20250724_111413_indicators.csv"
            path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=";")
        if "date" not in df.columns or "signal" not in df.columns:
            continue
        if closes is None and "close" in df.columns:
            closes = df[["date", "close"]].copy()
        df = df[["date", "signal"]].copy()
        df.rename(columns={"signal": f"signal_{strat}"}, inplace=True)
        all_signals.append(df)
    if not all_signals or closes is None:
        return []
    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), all_signals)
    merged = pd.merge(merged, closes, on="date", how="left")
    merged = merged.sort_values("date")
    sigcols = [c for c in merged.columns if c.startswith("signal_")]
    # Agregacja sygnałów
    if mode == "and":
        merged["signal"] = merged[sigcols].apply(
            lambda row: 1 if all(x == 1 for x in row) else -1 if all(x == -1 for x in row) else 0, axis=1)
    elif mode == "or":
        merged["signal"] = merged[sigcols].apply(
            lambda row: 1 if any(x == 1 for x in row) else -1 if any(x == -1 for x in row) else 0, axis=1)
    elif mode == "vote":
        weight_vals = [weights.get(c.replace("signal_", ""), 1.0) if weights else 1.0 for c in sigcols]
        def weighted_vote(row):
            s = sum(w * (row[col] if pd.notnull(row[col]) else 0) for w, col in zip(weight_vals, sigcols))
            if s > 0: return 1
            if s < 0: return -1
            return 0
        merged["signal"] = merged.apply(weighted_vote, axis=1)
    else:
        raise HTTPException(status_code=400, detail="Nieznany tryb agregacji.")
    # --- Oblicz equity curve ---
    cash = initial_cash
    position = 0  # 0-brak, 1-long, -1-short
    equity_curve = []
    last_price = None
    for idx, row in merged.iterrows():
        price = row["close"]
        signal = row["signal"]
        if pd.isnull(price) or pd.isnull(signal):
            equity_curve.append((row["date"], cash))
            continue
        # Prosty model: jeśli sygnał=1 i nie mamy pozycji – kup, jeśli sygnał=-1 i mamy long – sprzedaj na zero, short ignorujemy (lub zrób reverse, jak chcesz)
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 1:
            cash *= price / entry_price  # zaktualizuj equity
            position = 0
        equity_curve.append((row["date"], cash if position == 0 else cash * price / entry_price))
        last_price = price
    # Wyjście z pozycji na końcu
    if position == 1 and last_price:
        cash *= last_price / entry_price
    equity = pd.DataFrame(equity_curve, columns=["date", "equity"])
    return equity.to_dict(orient="records")

@app.get("/ping")
def ping():
    return {"ping": "pong"}
