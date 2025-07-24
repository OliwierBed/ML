import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_and_store_data(raw_dir="data/raw", processed_dir="data/processed", db_path="data/database.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tworzenie tabel
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            interval TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            interval TEXT,
            date TEXT,
            close REAL,
            adj_close REAL,
            close_norm REAL,
            sma_20 REAL,
            rsi_14 REAL,
            atr_14 REAL,
            obv REAL,
            bb_low REAL,
            bb_mid REAL,
            bb_high REAL,
            stoch_k REAL,
            stoch_d REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL
        )
    """)

    # Czyszczenie starych danych (starszych niż 2 lata)
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    cursor.execute("DELETE FROM stock_data WHERE date < ?", (two_years_ago,))
    cursor.execute("DELETE FROM indicators WHERE date < ?", (two_years_ago,))

    # Wstawianie danych z data/raw
    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            ticker, interval = file.split("_")[0], file.split("_")[1]
            logging.info(f"Wczytywanie {file}")
            try:
                df = pd.read_csv(os.path.join(raw_dir, file), sep=";")
                df["ticker"] = ticker
                df["interval"] = interval
                df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", 
                                  "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True)
                df.to_sql("stock_data", conn, if_exists="append", index=False)
                logging.info(f"Zapisano {file} do stock_data")
            except Exception as e:
                logging.error(f"Błąd przy wczytywaniu {file}: {str(e)}")

    # Wstawianie danych z data/processed
    indicator_columns = ["ticker", "interval", "date", "close", "adj_close", "close_norm", 
                        "sma_20", "rsi_14", "atr_14", "obv", "bb_low", "bb_mid", "bb_high", 
                        "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist"]
    
    for file in os.listdir(processed_dir):
        if file.endswith("_indicators.csv"):
            ticker, interval = file.split("_")[0], file.split("_")[1]
            logging.info(f"Wczytywanie {file}")
            try:
                df = pd.read_csv(os.path.join(processed_dir, file), sep=";")
                df["ticker"] = ticker
                df["interval"] = interval
                df.rename(columns={"Date": "date", "Close": "close", "Adj_Close": "adj_close", 
                                  "Close_Norm": "close_norm", "SMA_20": "sma_20", "RSI_14": "rsi_14", 
                                  "ATR_14": "atr_14", "OBV": "obv", "BB_Low": "bb_low", "BB_Mid": "bb_mid", 
                                  "BB_High": "bb_high", "Stoch_K": "stoch_k", "Stoch_D": "stoch_d", 
                                  "MACD": "macd", "MACD_Hist": "macd_hist", "MACD_Signal": "macd_signal"}, inplace=True)
                # Usuń nadmiarowe kolumny, np. High, Low, Open, Volume
                df = df[[col for col in indicator_columns if col in df.columns]]
                # Wypełnij NaN wartościami 0 dla kolumn numerycznych
                numeric_columns = ["close", "adj_close", "close_norm", "sma_20", "rsi_14", "atr_14", "obv", 
                                  "bb_low", "bb_mid", "bb_high", "stoch_k", "stoch_d", "macd", 
                                  "macd_signal", "macd_hist"]
                # Upewnij się, że wszystkie kolumny numeryczne istnieją
                for col in numeric_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df[numeric_columns] = df[numeric_columns].fillna(0)
                df.to_sql("indicators", conn, if_exists="append", index=False)
                logging.info(f"Zapisano {file} do indicators")
            except Exception as e:
                logging.error(f"Błąd przy wczytywaniu {file}: {str(e)}")

    conn.commit()
    conn.close()
    logging.info("Dane zapisane do bazy danych.")

if __name__ == "__main__":
    clean_and_store_data()