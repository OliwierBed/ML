import pandas as pd
import os
from collections import defaultdict
from config.load import load_config

config = load_config()

RESULTS_DIR = getattr(config.paths, "results", "data-pipelines/feature_stores/data/results")
PROCESSED_DIR = getattr(config.paths, "feature_stores_processed", "data-pipelines/feature_stores/data/processed")
ENSEMBLE_DIR = os.path.join(PROCESSED_DIR, "ensemble")
TOP_STRATEGIES_FILE = os.path.join(RESULTS_DIR, "top_per_bucket.csv")

os.makedirs(ENSEMBLE_DIR, exist_ok=True)

def find_backtest_file(filename, strategy, results_dir=RESULTS_DIR):
    # Zamien np. AAPL_1h_20250727_222034_indicators.csv -> AAPL_1h_20250727_222034_indicators_macd_backtest.csv
    base = filename.replace('.csv', f'_{strategy}_backtest.csv')
    path = os.path.join(results_dir, base)
    return path if os.path.exists(path) else None

def load_signal(file_path):
    df = pd.read_csv(file_path, sep=";")
    # sprawdź, czy mamy wymagane kolumny
    if "date" not in df.columns or "signal" not in df.columns:
        raise ValueError(f"Brak wymaganych kolumn 'date'/'signal' w {file_path}")
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

        file_path = find_backtest_file(file_name, strategy)
        if not file_path:
            print(f"❌ Nie znaleziono pliku sygnału: {file_name} [{strategy}]")
            continue

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
