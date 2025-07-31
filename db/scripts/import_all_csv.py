import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
import yaml
from datetime import datetime

# 🔁 Lokalna lista interwałów
INTERVALS = ["1h", "1d", "1wk"]

def infer_interval_from_filename(filename: str) -> str:
    for itv in INTERVALS:
        if f"_{itv}_" in filename:
            return itv
    return "1d"

def infer_ticker_from_filename(filename: str) -> str:
    return filename.split("_")[0]

# 📥 Wczytaj config
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=config["database"]["user"],
    password=config["database"]["password"],
    host=config["database"]["host"],
    port=config["database"]["port"],
    database=config["database"]["name"],
)
engine = create_engine(db_url)

RAW_DIR = config["data"]["raw_data_dir"]
PROCESSED_DIR = config["data"]["processed_data_dir"]

def insert_file_to_db(filepath, source="raw", strategy=None):
    filename = os.path.basename(filepath)
    interval = infer_interval_from_filename(filename)
    ticker = infer_ticker_from_filename(filename)

    df = pd.read_csv(filepath, sep=";")
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

    # Usuń błędne nagłówki i teksty w danych (np. "BB_High")
    df = df[~df.iloc[:, 0].astype(str).str.contains("date|open|close|bb_high|sma_signal", case=False)]

    # Popraw nazwę kolumny
    if "adj close" in df.columns:
        df.rename(columns={"adj close": "adj_close"}, inplace=True)

    # Znajdź kolumnę daty
    timestamp_col = None
    for cand in ["timestamp", "date", "datetime"]:
        if cand in df.columns:
            timestamp_col = cand
            break
    if timestamp_col is None:
        raise ValueError(f"Brak kolumny date lub timestamp w {filepath}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df[df[timestamp_col].notnull()]
    df["timestamp"] = df[timestamp_col]

    # Dodaj metadane
    df["ticker"] = ticker
    df["interval"] = interval
    df["source"] = source
    df["strategy"] = strategy
    df["is_latest"] = False
    df["meta"] = None
    df["inserted_at"] = datetime.utcnow()

    # Filtruj tylko dozwolone kolumny
    inspector = inspect(engine)
    db_columns = {col["name"] for col in inspector.get_columns("candles")}
    df = df[[col for col in df.columns if col in db_columns]]

    # Konwersja do typów numerycznych
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "ticker", "interval", "strategy"])

    df.to_sql("candles", engine, if_exists="append", index=False, chunksize=5000)
    print(f"✅ Wstawiono {len(df)} rekordów z {filepath} do DB.")

def run():
    for dirpath, _, filenames in os.walk(RAW_DIR):
        for fname in filenames:
            if fname.endswith(".csv"):
                fullpath = os.path.join(dirpath, fname)
                insert_file_to_db(fullpath, source="raw")

    for dirpath, _, filenames in os.walk(PROCESSED_DIR):
        for fname in filenames:
            if fname.endswith(".csv"):
                fullpath = os.path.join(dirpath, fname)
                insert_file_to_db(fullpath, source="processed")

if __name__ == "__main__":
    run()
