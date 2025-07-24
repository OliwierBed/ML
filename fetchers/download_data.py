import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import time
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(tickers=["AAPL", "MSFT", "TSLA"], intervals=["1d", "1h", "1wk"], days=730, days_1h=60, retries=3, proxy=None):
    # Ustaw proxy, jeśli podano
    if proxy:
        yf.set_config(proxy=proxy)
        logging.info(f"Using proxy: {proxy}")
    
    # Ustaw daty
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    start_date_1h = (datetime.now() - timedelta(days=days_1h)).strftime("%Y-%m-%d")
    
    os.makedirs("data/raw", exist_ok=True)

    for ticker in tickers:
        for interval in intervals:
            for attempt in range(retries):
                try:
                    start = start_date_1h if interval == "1h" else start_date
                    logging.info(f"Pobieranie danych dla {ticker} ({interval}), próba: {attempt + 1}/{retries}")
                    
                    df = yf.download(tickers=ticker, 
                                   start=start, 
                                   end=end_date, 
                                   interval=interval, 
                                   progress=False, 
                                   auto_adjust=False)
                    
                    if not df.empty:
                        # Sprawdź strukturę DataFrame
                        logging.debug(f"Kolumny DataFrame przed reset_index: {df.columns}")
                        df = df.reset_index()
                        
                        # Spłaszcz multi-indeks, jeśli istnieje
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                        
                        # Ensure Date column
                        if 'Date' not in df.columns and 'Datetime' in df.columns:
                            df.rename(columns={'Datetime': 'Date'}, inplace=True)
                        elif 'Date' not in df.columns:
                            df['Date'] = pd.to_datetime(df.index)
                        
                        logging.debug(f"Kolumny DataFrame po reset_index: {df.columns}")
                        
                        # Konwersja kolumn numerycznych
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
                        filepath = os.path.join("data/raw", filename)
                        df.to_csv(filepath, sep=";", decimal=".", index=False, encoding="utf-8-sig")
                        logging.info(f"Dane zapisane do {filepath}")
                        break
                    else:
                        logging.warning(f"Brak danych dla {ticker} ({interval})")
                        break
                except Exception as e:
                    logging.error(f"Nie udało się pobrać danych dla {ticker} ({interval}) w próbie {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        logging.info("Czekam 30 sekund przed kolejną próbą...")
                        time.sleep(30)
                    else:
                        logging.error(f"Rezygnuję po {retries} próbach dla {ticker} ({interval})")
        
        time.sleep(60)  # Opóźnienie między tickerami

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA", help="Comma-separated list of tickers")
    parser.add_argument("--intervals", type=str, default="1d,1h,1wk", help="Comma-separated list of intervals")
    parser.add_argument("--days", type=int, default=730, help="Number of days for 1d, 1wk")
    parser.add_argument("--days_1h", type=int, default=60, help="Number of days for 1h")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy server (e.g., http://your_proxy:port)")
    args = parser.parse_args()
    download_data(tickers=args.tickers.split(","), intervals=args.intervals.split(","), 
                  days=args.days, days_1h=args.days_1h, retries=args.retries, proxy=args.proxy)
