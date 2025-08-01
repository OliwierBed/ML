# backtest/runner_db.py

import os
import pandas as pd
from omegaconf import OmegaConf

from backtest.evaluate import evaluate_backtest
from backtest.portfolio import BacktestEngine
from db.utils.load_from_db import load_data_from_db
from backtest.utils import annualization_factor_from_interval

from backtest.strategies import (
    MACDCrossoverStrategy,
    RSIStrategy,
    SMAStrategy,
    EMAStrategy,
    BollingerBandsStrategy,
    EnsembleStrategy,
)


# 📄 Konfiguracja z pliku YAML
config = OmegaConf.load("config/config.yaml")
TICKERS = config["data"]["tickers"]
INTERVALS = config["data"]["intervals"]
STRATEGIES = config["backtest"]["strategies"]

# 🔁 Mapowanie nazw na klasy strategii
STRATEGY_MAP = {
    "macd": MACDCrossoverStrategy,
    "rsi": RSIStrategy,
    "sma": SMAStrategy,
    "ema": EMAStrategy,
    "bollinger": BollingerBandsStrategy,
    "ensemble": EnsembleStrategy,
}


def run_backtest(ticker, interval, strategy_name, initial_cash):
    df = load_data_from_db(ticker=ticker, interval=interval, columns=["close"])
    if df is None or df.empty:
        print(f"⛔ Brak danych: {ticker}, {interval}, {strategy_name}")
        return None

    strategy_class = STRATEGY_MAP.get(strategy_name)
    if strategy_class is None:
        print(f"⛔ Nieznana strategia: {strategy_name}")
        return None

    strategy = strategy_class(df)
    signals = strategy.generate_signals()

    if "signal" in df.columns:
        df.drop(columns=["signal"], inplace=True)
    df["signal"] = signals["signal"]

    if df["signal"].isna().all():
        print(f"⛔ Sygnał zawiera tylko NaN – pomijam {ticker} / {interval} / {strategy_name}")
        return None

    af = annualization_factor_from_interval(interval)
    bt = BacktestEngine(df, initial_cash=initial_cash, annualization_factor=af)
    results_df, metrics = bt.run(signal_col="signal")

    metrics["ticker"] = ticker
    metrics["interval"] = interval
    metrics["strategy"] = strategy_name
    return metrics


def run_batch_backtest():
    all_results = []
    for ticker in TICKERS:
        for interval in INTERVALS:
            for strategy_name in STRATEGIES:
                print(f"▶ Backtest: {ticker} / {interval} / {strategy_name}")
                metrics = run_backtest(
                    ticker,
                    interval,
                    strategy_name,
                    config["backtest"]["initial_cash"],
                )
                if metrics:
                    all_results.append(metrics)

    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs(config["paths"]["results"], exist_ok=True)
        output_path = os.path.join(config["paths"]["results"], "batch_backtest_results.csv")
        df.to_csv(output_path, index=False, sep=";")
        print(f"✅ Wyniki zapisane: {output_path}")
        return df
    else:
        raise ValueError("Brak wyników do zapisania.")


def run_backtest_for_api(tickers, intervals, strategies, initial_cash):
    all_results = []
    for ticker in tickers:
        for interval in intervals:
            for strategy in strategies:
                print(f"▶ Backtest: {ticker} / {interval} / {strategy}")
                metrics = run_backtest(ticker, interval, strategy, initial_cash)
                if metrics:
                    all_results.append(metrics)
    return all_results
