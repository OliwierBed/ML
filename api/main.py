from fastapi import FastAPI, Query, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from functools import reduce
from api.routers import ml
from api.routers import backtest
from sqlalchemy.orm import Session
from db.session import get_db
from db.models import Candle
from db.utils.load_from_db import load_data_from_db

from backtest.strategies import (
    MACDCrossoverStrategy,
    RSIStrategy,
    SMAStrategy,
    EMAStrategy,
    BollingerBandsStrategy,
)

STRATEGY_MAP = {
    "macd": MACDCrossoverStrategy,
    "rsi": RSIStrategy,
    "sma": SMAStrategy,
    "ema": EMAStrategy,
    "bollinger": BollingerBandsStrategy,
}

app = FastAPI(title="TradingBot API")

app.include_router(ml.router)
app.include_router(backtest.router)  # 👈

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = "data-pipelines/feature_stores/data/results"


def _load_signals(db: Session, ticker: str, interval: str, strategy: str) -> pd.DataFrame:
    source = "ensemble" if strategy == "ensemble" else "processed"
    try:
        q = db.query(Candle.timestamp.label("date"), Candle.signal, Candle.close).filter(
            Candle.ticker == ticker,
            Candle.interval == interval,
            Candle.source == source,
        )
        if strategy != "ensemble":
            q = q.filter(Candle.strategy == strategy)
        q = q.order_by(Candle.timestamp.asc())
        df = pd.read_sql(q.statement, db.bind)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        raw = load_data_from_db(ticker=ticker, interval=interval, columns=["close"])
        if raw.empty or strategy == "ensemble":
            return raw
        strat_cls = STRATEGY_MAP.get(strategy)
        if strat_cls is None:
            return pd.DataFrame()
        raw = raw.copy()
        if "date" not in raw.columns and "timestamp" in raw.columns:
            raw.rename(columns={"timestamp": "date"}, inplace=True)
        raw["date"] = pd.to_datetime(raw["date"])
        signals = strat_cls(raw).generate_signals()
        raw["signal"] = signals["signal"].values
        return raw[["date", "signal", "close"]]

    df["date"] = pd.to_datetime(df["date"])
    return df


def _list_strategies(db: Session) -> list[str]:
    """Return all strategy names registered in the database."""
    strategies = (
        db.query(Candle.strategy)
        .filter(Candle.strategy.isnot(None))
        .distinct()
        .all()
    )
    strat_list = sorted({s[0] for s in strategies if s[0]})
    if "ensemble" not in strat_list:
        strat_list.append("ensemble")
    return strat_list

@app.get("/tickers")
def get_tickers(db: Session = Depends(get_db)):
    try:
        tickers = (
            db.query(Candle.ticker)
            .filter(Candle.source == "raw")
            .distinct()
            .all()
        )
        return {"tickers": sorted(t[0] for t in tickers)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intervals")
def get_intervals(db: Session = Depends(get_db)):
    try:
        intervals = (
            db.query(Candle.interval)
            .filter(Candle.source == "raw")
            .distinct()
            .all()
        )
        return {"intervals": sorted(i[0] for i in intervals)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies")
def get_strategies(db: Session = Depends(get_db)):
    try:
        return {"strategies": _list_strategies(db)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    strategy: str = "ensemble",
    db: Session = Depends(get_db),
):
    df = _load_signals(db, ticker, interval, strategy)
    if df.empty:
        raise HTTPException(status_code=404, detail="Brak sygnałów")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = [c for c in df.columns if c in ("date", "signal", "close")]
    return df[cols].to_dict(orient="records")

# --- NOWOŚĆ: endpoint do agregacji sygnałów ---
@app.post("/signals/aggregate")
def aggregate_signals(body: dict = Body(...), db: Session = Depends(get_db)):
    ticker = body.get("ticker")
    interval = body.get("interval")
    strategies = body.get("strategies")
    mode = body.get("mode", "and")
    weights = body.get("weights", None)
    if not (ticker and interval and mode):
        raise HTTPException(status_code=400, detail="Brak wymaganych parametrów.")
    if not strategies:
        strategies = _list_strategies(db)

    all_signals = []
    for strat in strategies:
        df = _load_signals(db, ticker, interval, strat)
        if df.empty:
            continue
        df = df[["date", "signal"]].copy()
        df.rename(columns={"signal": f"signal_{strat}"}, inplace=True)
        all_signals.append(df)
    if not all_signals:
        return []
    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), all_signals)
    merged = merged.sort_values("date")
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
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return merged[["date", "signal"]].to_dict(orient="records")

# --- NOWOŚĆ: endpoint do equity curve dla agregatu ---
@app.post("/equity/aggregate")
def aggregate_equity(body: dict = Body(...), db: Session = Depends(get_db)):
    ticker = body.get("ticker")
    interval = body.get("interval")
    strategies = body.get("strategies")
    mode = body.get("mode", "and")
    weights = body.get("weights", None)
    initial_cash = 100000  # albo z configu

    if not strategies:
        strategies = _list_strategies(db)

    all_signals = []
    closes = None
    for strat in strategies:
        df = _load_signals(db, ticker, interval, strat)
        if df.empty:
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

    cash = initial_cash
    position = 0
    equity_curve = []
    last_price = None
    for _, row in merged.iterrows():
        price = row["close"]
        signal = row["signal"]
        if pd.isnull(price) or pd.isnull(signal):
            equity_curve.append((row["date"], cash))
            continue
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 1:
            cash *= price / entry_price
            position = 0
        equity_curve.append((row["date"], cash if position == 0 else cash * price / entry_price))
        last_price = price
    if position == 1 and last_price:
        cash *= last_price / entry_price
    equity = pd.DataFrame(equity_curve, columns=["date", "equity"])
    equity["date"] = pd.to_datetime(equity["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return equity.to_dict(orient="records")

@app.get("/ping")
def ping():
    return {"ping": "pong"}
