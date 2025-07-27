import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml

from backtest.rules import get_strategy
from backtest.portfolio import BacktestEngine
from backtest.evaluate import evaluate_backtest

CONFIG_PATH = "config/config.yaml"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_processed_csv(path):
    df = pd.read_csv(path, sep=";", parse_dates=["date"])
    df.columns = [col.lower() for col in df.columns]
    return df

def save_metrics(metrics: dict, save_path: Path):
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, sep=";", index=False)

def main():
    config = load_config(CONFIG_PATH)

    processed_path = Path(config["paths"]["feature_store_processed"])
    results_path = Path(config["paths"]["results"])
    results_path.mkdir(parents=True, exist_ok=True)

    strategies = config["backtest"]["strategies"]
    initial_cash = config["backtest"]["initial_cash"]

    print(f"📁 Używam katalogu z danymi: {processed_path}")
    print(f"🧠 Strategie: {strategies}")

    csv_files = list(processed_path.glob("*.csv"))
    if not csv_files:
        print("⚠️ Nie znaleziono żadnych plików CSV.")
        return

    for file in tqdm(csv_files, desc="🔄 Przetwarzanie plików"):
        ticker = file.name.split("_")[0]
        timeframe = file.name.split("_")[1]

        for strategy_name in strategies:
            try:
                df = load_processed_csv(file)
                strategy = get_strategy(strategy_name, df)
                strategy.generate_signals()

                engine = BacktestEngine(
                    df=df,
                    signal_column="signal",
                    initial_cash=initial_cash
                )
                engine.run()

                metrics = evaluate_backtest(engine)

                output_filename = f"{ticker}_{timeframe}_{strategy_name}_metrics.csv"
                save_path = results_path / output_filename
                save_metrics(metrics, save_path)

                print(f"✅ Zapisano wyniki: {save_path.name}")

            except Exception as e:
                print(f"❌ Błąd przy pliku {file.name} [{strategy_name}]: {e}")

if __name__ == "__main__":
    main()
