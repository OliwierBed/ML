import os
import pandas as pd

PROCESSED_DIR = "data-pipelines/feature_stores/data/processed"

def load_series(ticker: str, interval: str, cols=("Close",)):
    # weź przetworzone z wskaźnikami, żeby ML dev mógł korzystać z większej liczby cech
    # (możesz też dodać path do ensemble_full jeśli chcesz)
    fn = [f for f in os.listdir(PROCESSED_DIR)
          if f.startswith(f"{ticker}_{interval}_") and f.endswith("_indicators.csv")]
    if not fn:
        raise FileNotFoundError(f"Brak plików processed dla {ticker} {interval}")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, fn[0]), sep=";")
    return df
