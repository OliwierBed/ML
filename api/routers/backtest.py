from fastapi import APIRouter, HTTPException
from backtest.runner_db import run_backtest_for_api, STRATEGY_MAP

import yaml
import os

router = APIRouter()

@router.get("/backtest/run")
def run_backtest_endpoint():
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tickers = config["data"]["tickers"]
        intervals = config["data"]["intervals"]
        strategies = [s for s in config["backtest"]["strategies"] if s in STRATEGY_MAP]
        initial_cash = config["backtest"]["initial_cash"]

        results = run_backtest_for_api(tickers, intervals, strategies, initial_cash)
        if not results:
            raise HTTPException(status_code=404, detail="Brak wyników backtestu")
        return {"status": "ok", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

