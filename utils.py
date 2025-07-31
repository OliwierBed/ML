import os
import pandas as pd
from sqlalchemy import create_engine
import yaml
from datetime import datetime


def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def insert_file_to_db(filepath, source=None, strategy=None):
    config = load_config()
    db_url = f"postgresql+psycopg2://{config['database']['user']}:{config['database']['password']}@" \
             f"{config['database']['host']}:{config['database']['port']}/{config['database']['name']}"
    engine = create_engine(db_url)

    filename = os.path.basename(filepath)
    ticker, interval, *_ = filename.split("_")
    interval = interval.strip().lower()

    df = pd.read_csv(filepath, sep=";")
    df.columns = [col.lower() for col in df.columns]  # <- normalizacja nagłówków

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"Brak kolumny date lub timestamp w {filepath}")

    df["ticker"] = ticker
    df["interval"] = interval
    df["source"] = source
    df["strategy"] = strategy
    df["is_latest"] = False
    df["meta"] = None

    df.to_sql("candles", engine, if_exists="append", index=False)
    print(f"✅ Zaimportowano {filepath} ({len(df)} rekordów)")

def run():
    config = load_config()
    raw_data_dir = config["data"]["raw_data_dir"]
    sources = ["raw", "processed", "ensemble"]

    for source in sources:
        source_path = os.path.join(raw_data_dir, source)
        if not os.path.exists(source_path):
            continue

        for filename in os.listdir(source_path):
            if not filename.endswith(".csv"):
                continue

            fullpath = os.path.join(source_path, filename)
            try:
                if "ensemble" in fullpath or "processed" in fullpath:
                    strategy = filename.split("_")[0].lower()
                    insert_file_to_db(fullpath, source=source, strategy=strategy)
                else:
                    insert_file_to_db(fullpath, source=source)
            except Exception as e:
                print(f"❌ Błąd podczas importu {filename}: {e}")

if __name__ == "__main__":
    run()
