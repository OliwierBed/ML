import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_db(db_path="data/database.db"):
    try:
        conn = sqlite3.connect(db_path)
        # Sprawdź schemat tabeli indicators
        df_schema = pd.read_sql_query("PRAGMA table_info(indicators)", conn)
        logging.info("Schemat tabeli indicators:")
        logging.info(df_schema.to_string())
        
        # Sprawdź dane dla AAPL 1d
        df_data = pd.read_sql_query("SELECT * FROM indicators WHERE ticker='AAPL' AND interval='1d' LIMIT 10", conn)
        logging.info("Dane dla AAPL (1d):")
        logging.info(df_data.to_string())
        
        # Sprawdź dane dla AAPL 1h
        df_data_h = pd.read_sql_query("SELECT * FROM indicators WHERE ticker='AAPL' AND interval='1h' LIMIT 10", conn)
        logging.info("Dane dla AAPL (1h):")
        logging.info(df_data_h.to_string())
        
        # Sprawdź, czy są zera
        df_zeros = pd.read_sql_query("SELECT COUNT(*) as zero_count FROM indicators WHERE sma_20 = 0 OR rsi_14 = 0 OR macd = 0", conn)
        logging.info(f"Liczba wierszy z zerami w sma_20, rsi_14 lub macd: {df_zeros['zero_count'][0]}")
        
        conn.close()
    except Exception as e:
        logging.error(f"Błąd weryfikacji bazy: {str(e)}")

if __name__ == "__main__":
    verify_db()