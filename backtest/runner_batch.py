import os
import pandas as pd
from backtest.portfolio import BacktestEngine

# 👉 Możliwe do przeniesienia później do config.yaml
TICKERS = ["AAPL", "MSFT", "TSLA"]
INTERVALS = ["1h", "1d", "1wk"]
PROCESSED_DIR = "data-pipelines/feature_stores/data/processed"

results = []

for filename in os.listdir(PROCESSED_DIR):
    if not filename.endswith(".csv"):
        continue

    if not any(ticker in filename for ticker in TICKERS):
        continue

    if not any(interval in filename for interval in INTERVALS):
        continue

    print(f"🔁 Przetwarzam: {filename}")
    path = os.path.join(PROCESSED_DIR, filename)

    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [col.lower() for col in df.columns]

        # 🧠 Znajdź i przekształć kolumnę daty
        date_col = next((col for col in df.columns if "date" in col), None)
        if date_col is None:
            raise ValueError("Brak kolumny daty")

        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

        if "signal" not in df.columns:
            print(f"⚠️  Pominięto {filename} – brak kolumny 'signal'")
            continue

        bt = BacktestEngine(df)
        stats = bt.run()
        stats["filename"] = filename
        stats["interval"] = next((i for i in INTERVALS if i in filename), "N/A")
        results.append(stats)

    except Exception as e:
        print(f"❌ Błąd przy pliku {filename}: {e}")

if results:
    summary = pd.DataFrame(results)
    summary = summary.sort_values(by="sharpe", ascending=False)
    print("\n📊 Wyniki zbiorcze (posortowane po Sharpe):")
    print(summary.to_string(index=False))
else:
    print("⚠️  Nie znaleziono żadnych prawidłowych plików do przetworzenia.")
