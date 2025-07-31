from fastapi import APIRouter
from backtest.runner_db import run_backtest_for_api

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
        strategies = config["backtest"]["strategies"]
        initial_cash = config["backtest"]["initial_cash"]

        results = run_backtest_for_api(tickers, intervals, strategies, initial_cash)
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

