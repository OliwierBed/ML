# backtest/runner_ensemble.py

import os
import pandas as pd
from omegaconf import OmegaConf

from backtest.utils import (
    load_processed_csv_lower,
    infer_interval_from_filename,
    annualization_factor_from_interval,
)
from backtest.portfolio import BacktestEngine

def main():
    config = OmegaConf.load("config/config.yaml")
    ENSEMBLE_DIR = os.path.join(config.paths.feature_stores_processed, "ensemble")
    RESULTS_DIR = config.paths.results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    INITIAL_CASH = float(config.backtest.initial_cash)

    print(f"📁 Używam katalogu ENSEMBLE: {ENSEMBLE_DIR}\n")
    all_results = []

    # Szukaj tylko plików *_ensemble_full.csv
    files = [f for f in os.listdir(ENSEMBLE_DIR) if f.endswith("_ensemble_full.csv")]
    if not files:
        print("⚠️ Nie znaleziono żadnych plików *_ensemble_full.csv do przetworzenia.")
        return

    for filename in files:
        ticker = filename.split("_")[0].upper()
        interval = infer_interval_from_filename(filename)
        af = annualization_factor_from_interval(interval)
        print(f"🔁 Przetwarzam: {filename}")

        full_path = os.path.join(ENSEMBLE_DIR, filename)
        try:
            df = load_processed_csv_lower(full_path)
            if "close" not in df.columns:
                raise ValueError("Brak kolumny 'close' w danych!")

            # Zakładamy, że ensemble zawsze używa 'signal' jako kolumny wejściowej
            if "signal" not in df.columns:
                raise ValueError("Brak kolumny 'signal' (ensemble) w danych!")

            engine = BacktestEngine(df, initial_cash=INITIAL_CASH, annualization_factor=af)
            results_df, metrics = engine.run(signal_col="signal")

            # Doklej meta
            metrics["ticker"] = ticker
            metrics["interval"] = interval
            metrics["strategy"] = "ensemble"
            metrics["filename"] = filename
            all_results.append(metrics)

            # Zapisz szczegółowe wyniki (opcjonalnie)
            out_csv = os.path.join(
                RESULTS_DIR,
                f"{filename.replace('.csv', '')}_ensemble_backtest.csv",
            )
            results_df.to_csv(out_csv, sep=";")
        except Exception as e:
            print(f"❌ Błąd przy pliku {filename}: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        summary = summary.sort_values(by="sharpe", ascending=False)

        print("\n📊 Wyniki ENSEMBLE (posortowane po Sharpe):")
        with pd.option_context("display.float_format", "{:.6f}".format):
            print(summary.to_string(index=False))

        out_summary = os.path.join(RESULTS_DIR, "ensemble_batch_results.csv")
        summary.to_csv(out_summary, index=False, sep=";")
        print(f"\n💾 Zapisano zbiorcze wyniki ensemble do: {out_summary}")
    else:
        print("\n⚠️ Nie znaleziono żadnych prawidłowych plików ensemble do przetworzenia.")

if __name__ == "__main__":
    main()
