import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import time
import logging
import yaml

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_data(
    tickers=None, intervals=None, days=730, days_1h=60, retries=3, proxy=None, raw_data_dir=None
):
    config = load_config()
    if raw_data_dir is None:
        raw_data_dir = config["data"].get("raw_data_dir", "data-pipelines/feature_stores/data/raw")
    if tickers is None:
        tickers = config["data"].get("tickers", ["AAPL", "MSFT", "TSLA"])
    if intervals is None:
        intervals = config["data"].get("intervals", ["1d", "1h", "1wk"])

    if proxy:
        yf.set_config(proxy=proxy)
        logging.info(f"Using proxy: {proxy}")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    start_date_1h = (datetime.now() - timedelta(days=days_1h)).strftime("%Y-%m-%d")
    
    os.makedirs(raw_data_dir, exist_ok=True)

    for ticker in tickers:
        for interval in intervals:
            for attempt in range(retries):
                try:
                    start = start_date_1h if interval == "1h" else start_date
                    logging.info(f"Pobieranie danych dla {ticker} ({interval}), próba: {attempt + 1}/{retries}")
                    
                    df = yf.download(
                        tickers=ticker, 
                        start=start, 
                        end=end_date, 
                        interval=interval, 
                        progress=False, 
                        auto_adjust=False
                    )
                    
                    if not df.empty:
                        df = df.reset_index()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                        if 'Date' not in df.columns and 'Datetime' in df.columns:
                            df.rename(columns={'Datetime': 'Date'}, inplace=True)
                        elif 'Date' not in df.columns:
                            df['Date'] = pd.to_datetime(df.index)

                        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                        for col in numeric_cols:
                            if col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col], errors="coerce")
                                    df = df.dropna(subset=[col])
                                except Exception as e:
                                    logging.error(f"Błąd konwersji kolumny {col} dla {ticker} ({interval}): {e}")
                                    raise
                        if df.empty:
                            logging.warning(f"Brak poprawnych danych dla {ticker} ({interval})")
                            continue
                        filename = f"{ticker}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        filepath = os.path.join(raw_data_dir, filename)
                        df.to_csv(filepath, sep=";", decimal=".", index=False, encoding="utf-8-sig")
                        logging.info(f"Dane zapisane do {filepath}")
                        break
                    else:
                        logging.warning(f"Brak danych dla {ticker} ({interval})")
                        break
                except Exception as e:
                    logging.error(f"Nie udało się pobrać danych dla {ticker} ({interval}) w próbie {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        logging.info("Retry za chwilę...")
                        time.sleep(0.2)  # minimalne opóźnienie
                    else:
                        logging.error(f"Rezygnuję po {retries} próbach dla {ticker} ({interval})")
        # time.sleep(1)  # <- usunięte opóźnienie między tickerami

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers")
    parser.add_argument("--intervals", type=str, default=None, help="Comma-separated list of intervals")
    parser.add_argument("--days", type=int, default=730, help="Number of days for 1d, 1wk")
    parser.add_argument("--days_1h", type=int, default=60, help="Number of days for 1h")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy server (e.g., http://your_proxy:port)")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Directory for raw data")
    args = parser.parse_args()
    tickers = args.tickers.split(",") if args.tickers else None
    intervals = args.intervals.split(",") if args.intervals else None
    download_data(
        tickers=tickers,
        intervals=intervals,
        days=args.days,
        days_1h=args.days_1h,
        retries=args.retries,
        proxy=args.proxy,
        raw_data_dir=args.raw_data_dir
    )
