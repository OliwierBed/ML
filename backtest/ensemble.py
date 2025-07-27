import pandas as pd
import os
from collections import defaultdict
from config.load import load_config

config = load_config()

TOP_STRATEGIES_FILE = "backtest/results/top_per_bucket.csv"
PROCESSED_DIR = config.paths.feature_stores_processed
ENSEMBLE_DIR = os.path.join(config.paths.feature_stores_processed, "ensemble")

os.makedirs(ENSEMBLE_DIR, exist_ok=True)

def load_signal(file_path):
    df = pd.read_csv(file_path, sep=";")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["signal"]

def main():
    top_df = pd.read_csv(TOP_STRATEGIES_FILE)

    grouped_signals = defaultdict(list)
    index_reference = {}

    for _, row in top_df.iterrows():
        file_name = row["filename"]
        ticker = row["ticker"]
        interval = row["interval"]
        strategy = row["strategy"]

        file_path = os.path.join(PROCESSED_DIR, file_name)

        try:
            signal_series = load_signal(file_path)
        except Exception as e:
            print(f"❌ Błąd przy ładowaniu {file_path}: {e}")
            continue

        key = (ticker, interval)
        grouped_signals[key].append(signal_series)

        # zapisz indeks czasowy dla danego klucza (wystarczy raz)
        if key not in index_reference:
            index_reference[key] = signal_series.index

    for (ticker, interval), signals in grouped_signals.items():
        if len(signals) == 0:
            print(f"⚠️ Brak sygnałów dla {ticker}, {interval}")
            continue

        # Zrównaj długości
        signals = [s.reindex(index_reference[(ticker, interval)]).fillna(0) for s in signals]
        df_signals = pd.concat(signals, axis=1)

        # Średnia i próg
        ensemble_signal = (df_signals.mean(axis=1) > 0.5).astype(int)

        # Zapisz do pliku
        output_df = pd.DataFrame({
            "date": index_reference[(ticker, interval)],
            "signal": ensemble_signal.values
        })

        output_path = os.path.join(ENSEMBLE_DIR, f"{ticker}_{interval}_ensemble.csv")
        output_df.to_csv(output_path, sep=";", index=False)
        print(f"✅ Zapisano ensemble dla {ticker} {interval} ➜ {output_path}")

    print("🏁 Ensemble strategii zakończony.")

if __name__ == "__main__":
    main()
