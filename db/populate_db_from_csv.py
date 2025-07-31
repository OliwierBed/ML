import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, Candle
from datetime import datetime
import yaml

# Wczytaj konfigurację DB
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_cfg = config["database"]
DATABASE_URL = f"postgresql://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def insert_candles_from_csv(file_path, ticker, interval, source="raw", strategy=None):
    df = pd.read_csv(file_path, sep=";")
    df.columns = [col.lower() for col in df.columns]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["created_at"] = datetime.utcnow()
    df.dropna(subset=["date"], inplace=True)

    if source == "raw":
        required_cols = ["open", "high", "low", "close", "volume"]
        df.dropna(subset=required_cols, inplace=True)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df.dropna(subset=["volume"], inplace=True)
    elif source == "processed":
        if "signal" not in df.columns:
            print(f"[⚠] Pomijam {file_path} – brak kolumny 'signal'")
            return
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

    # Usuń istniejące rekordy (jeśli są)
    delete_query = session.query(Candle).filter_by(
        ticker=ticker,
        interval=interval,
        source=source,
        strategy=strategy
    )
    delete_query.delete()
    session.commit()

    # Tworzenie obiektów
    candles = []
    for _, row in df.iterrows():
        candles.append(Candle(
            ticker=ticker,
            interval=interval,
            date=row["date"],
            open=float(row["open"]) if "open" in row and pd.notna(row["open"]) else None,
            high=float(row["high"]) if "high" in row and pd.notna(row["high"]) else None,
            low=float(row["low"]) if "low" in row and pd.notna(row["low"]) else None,
            close=float(row["close"]) if "close" in row and pd.notna(row["close"]) else None,
            volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
            signal=float(row["signal"]) if "signal" in row and pd.notna(row["signal"]) else None,
            source=source,
            strategy=strategy,
            created_at=datetime.utcnow(),
        ))

    session.bulk_save_objects(candles)
    session.commit()
    print(f"[✓] Wstawiono {len(candles)} rekordów z {file_path}")



def run():
    raw_dir = config["data"]["raw_data_dir"]
    processed_dir = config["data"]["processed_data_dir"]

    # Przetwórz raw
    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            path = os.path.join(raw_dir, file)
            parts = file.replace(".csv", "").split("_")
            ticker = parts[0]
            interval = parts[1]
            insert_candles_from_csv(path, ticker, interval, source="raw")

    # Przetwórz processed
    for strategy in os.listdir(processed_dir):
        strat_dir = os.path.join(processed_dir, strategy)
        if not os.path.isdir(strat_dir):
            continue
        for file in os.listdir(strat_dir):
            if file.endswith(".csv"):
                path = os.path.join(strat_dir, file)
                parts = file.replace(".csv", "").split("_")
                ticker = parts[0]
                interval = parts[1]
                insert_candles_from_csv(path, ticker, interval, source="processed", strategy=strategy)

if __name__ == "__main__":
    Base.metadata.create_all(engine)  # tworzy tabele, jeśli nie istnieją
    run()
