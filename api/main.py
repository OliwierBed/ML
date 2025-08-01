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
from config.load import load_config

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

CONFIG = load_config()
DEFAULT_TICKERS = list(CONFIG.data.tickers)
DEFAULT_INTERVALS = list(CONFIG.data.intervals)
DEFAULT_STRATEGIES = list(CONFIG.backtest.strategies)

app = FastAPI(title="TradingBot API")
app.include_router(ml.router)
app.include_router(backtest.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG = CONFIG
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
        try:
            raw = load_data_from_db(ticker=ticker, interval=interval, columns=["close"])
        except Exception:
            return pd.DataFrame()
        if raw.empty or strategy == "ensemble":
            return pd.DataFrame()
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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "signal" not in df.columns:
        return pd.DataFrame()
    return df

def _list_strategies(db: Session) -> list[str]:
    try:
        strategies = (
            db.query(Candle.strategy)
            .filter(Candle.strategy.isnot(None))
            .distinct()
            .all()
        )
        strat_list = sorted({s[0] for s in strategies if s[0]})
    except Exception:
        strat_list = []
    if not strat_list:
        strat_list = list(DEFAULT_STRATEGIES)
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
        tickers = sorted(t[0] for t in tickers if t[0])
    except Exception:
        tickers = []
    if not tickers:
        tickers = DEFAULT_TICKERS
    return {"tickers": tickers}

@app.get("/intervals")
def get_intervals(db: Session = Depends(get_db)):
    try:
        intervals = (
            db.query(Candle.interval)
            .filter(Candle.source == "raw")
            .distinct()
            .all()
        )
        intervals = sorted(i[0] for i in intervals if i[0])
    except Exception:
        intervals = []
    if not intervals:
        intervals = DEFAULT_INTERVALS
    return {"intervals": intervals}

@app.get("/strategies")
def get_strategies(db: Session = Depends(get_db)):
    try:
        strategies = _list_strategies(db)
    except Exception:
        strategies = list(DEFAULT_STRATEGIES) + ["ensemble"]
    return {"strategies": strategies}
