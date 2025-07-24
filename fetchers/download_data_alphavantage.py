import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import time
import logging
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# Wczytaj zmienne środowiskowe z pliku .env
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(tickers=["AAPL", "MSFT"], intervals=["1d", "1h", "1wk"], days=730, days_1h=60, retries=3):
    # Pobierz klucz API z pliku .env
    api_key = os.getenv("ALPHA_VANTAGE_API")
    if not api_key:
        logging.error("Brak klucza API w pliku .env (ALPHA_VANTAGE_API)")
        raise ValueError("Brak klucza API w pliku .env (ALPHA_VANTAGE_API)")
    
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    # Ustaw daty w formacie YYYY-MM-DD
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    start_date_1h = (datetime.now() - timedelta(days=days_1h)).strftime("%Y-%m-%d")
    
    os.makedirs("data/raw", exist_ok=True)

    interval_map = {
        "1d": "daily_adjusted",
        "1h": "intraday_60min",
        "1wk": "weekly_adjusted"
    }

    for ticker in tickers:
        for interval in intervals:
            for attempt in range(retries):
                try:
                    start = start_date_1h if interval == "1h" else start_date
                    logging.info(f"Pobieranie danych dla {ticker} ({interval}), start: {start}, end: {end_date}, próba: {attempt + 1}/{retries}")
                    
                    if interval == "1h":
                        df, _ = ts.get_intraday(symbol=ticker, interval="60min", outputsize="full")
                    elif interval == "1wk":
                        df, _ = ts.get_weekly_adjusted(symbol=ticker)
                    else:  # 1d
                        df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
                    
                    if not df.empty:
                        df = df.reset_index()
                        # Dopasuj nazwy kolumn do formatu yfinance
                        df.rename(columns={
                            "date": "Date",
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. adjusted close": "Adj Close",
                            "6. volume": "Volume"
                        }, inplace=True)
                        
                        # Filtruj daty
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df[(df['Date'] >= start) & (df['Date'] <= end_date)]
                        
                        # Usuń wiersze z wartościami tekstowymi
                        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                                df = df[df[col].notna()]
                        
                        if df.empty:
                            logging.warning(f"No valid data after cleaning for {ticker} ({interval})")
                            continue

                        filename = f"{ticker}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        filepath = os.path.join("data/raw", filename)
                        df.to_csv(filepath, sep=";", decimal=".", index=False, encoding="utf-8-sig")
                        logging.info(f"Data downloaded for {ticker} ({interval}) to {filepath}")
                        break  # Sukces, wychodzimy z pętli prób
                    else:
                        logging.warning(f"No data for {ticker} ({interval})")
                        break
                except Exception as e:
                    logging.error(f"Failed to download {ticker} ({interval}) on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        logging.info(f"Waiting 15 seconds before retrying...")
                        time.sleep(15)  # Limit Alpha Vantage: 5 żądań/minutę
                    else:
                        logging.error(f"Giving up after {retries} attempts for {ticker} ({interval})")
        
        # Opóźnienie między tickerami (limit Alpha Vantage: 5 żądań na minutę)
        time.sleep(15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data with Alpha Vantage")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA", help="Comma-separated list of tickers")
    parser.add_argument("--intervals", type=str, default="1d,1h,1wk", help="Comma-separated list of intervals (1d,1h,1wk)")
    parser.add_argument("--days", type=int, default=730, help="Number of days to download for 1d, 1wk")
    parser.add_argument("--days_1h", type=int, default=60, help="Number of days to download for 1h")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed downloads")
    args = parser.parse_args()
    download_data(tickers=args.tickers.split(","), intervals=args.intervals.split(","), 
                  days=args.days, days_1h=args.days_1h)