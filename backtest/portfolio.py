import os
import glob
import pandas as pd
from backtest.rules import get_strategy
from backtest.portfolio import BacktestEngine
from backtest.evaluate import evaluate_backtest
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_dir = config["paths"]["feature_store_processed"]
results_dir = config["paths"]["results"]
strategies = config["backtest"]["strategies"]
initial_cash = config["backtest"]["initial_cash"]

print(f">>\n📁 Używam katalogu z danymi: {data_dir}")
print(f"🧠 Strategie: {strategies}\n")

os.makedirs(results_dir, exist_ok=True)
results = []

for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
    df = pd.read_csv(filepath, sep=";")
    filename = os.path.basename(filepath)

    for strategy_name in strategies:
        print(f"🔁 Przetwarzam: {filename} [{strategy_name}]")
        try:
            StrategyClass = get_strategy(strategy_name)
            strategy = StrategyClass(df)
            df_signals = strategy.generate_signals()

            bt = BacktestEngine(df_signals, initial_cash=initial_cash)
            result_df = bt.run()
            metrics = evaluate_backtest(result_df, initial_cash)

            interval = filename.split("_")[1]
            results.append({
                **metrics,
                "filename": filename,
                "interval": interval,
                "strategy": strategy_name,
            })

        except Exception as e:
            print(f"❌ Błąd przy pliku {filename} [{strategy_name}]: {e}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe", ascending=False)
results_path = os.path.join(results_dir, "batch_results.csv")
results_df.to_csv(results_path, index=False)

print(f"\n📊 Wyniki zbiorcze zapisane do: {results_path}")
