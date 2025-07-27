import os
import pandas as pd

DATA_DIR = "data-pipelines/feature_stores/data/processed"

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Zamienia nazwy kolumn na małe litery, by ustandaryzować nazewnictwo."""
    df.columns = [col.lower() for col in df.columns]
    return df

def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumnę 'signal' w oparciu o logikę przecięcia MACD."""
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        print("⚠️  Pominięto – brak kolumn 'macd' i 'macd_signal'")
        return df

    df["signal"] = 0
    df.loc[df["macd"] > df["macd_signal"], "signal"] = 1
    df.loc[df["macd"] < df["macd_signal"], "signal"] = -1
    return df

def main():
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path, sep=";", parse_dates=["Date"])
        df = standardize_column_names(df)

        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df = generate_signal(df)
            df.to_csv(path, sep=";", index=False)
            print(f"✅ Dodano signal → {file}")
        else:
            print(f"⚠️  Pominięto {file} – brak MACD/MACD_Signal")

if __name__ == "__main__":
    main()
