import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import time
import logging
from yahoo_fin.stock_info import get_data

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(tickers=["AAPL", "MSFT", "TSLA"], intervals=["1d", "1h", "1wk"], days=730, days_1h=60):
    # Ustaw daty w formacie YYYY-MM-DD
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    start_date_1h = (datetime.now() - timedelta(days=days_1h)).strftime("%Y-%m-%d")
    
    os.makedirs("data/raw", exist_ok=True)

    for ticker in tickers:
        for interval in intervals:
            try:
                start = start_date_1h if interval == "1h" else start_date
                logging.info(f"Pobieranie danych dla {ticker} ({interval}), start: {start}, end: {end_date}")
                
                # Mapowanie interwałów yahoo_fin na yfinance
                interval_map = {"1d": "1d", "1h": "1h", "1wk": "1wk"}
                df = get_data(ticker, 
                            start_date=start, 
                            end_date=end_date, 
                            interval=interval_map[interval])
                
                if not df.empty:
                    df = df.reset_index()
                    # Ensure Date column exists
                    if 'Date' not in df.columns:
                        if 'Datetime' in df.columns:
                            df.rename(columns={'Datetime': 'Date'}, inplace=True)
                        else:
                            df.insert(0, 'Date', pd.to_datetime(df.index))
                    
                    # Usuń wiersze z wartościami tekstowymi w kolumnach numerycznych
                    numeric_cols = ["open", "high", "low", "close", "adjclose", "volume"]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            df = df[df[col].notna()]
                    
                    if df.empty:
                        logging.warning(f"No valid data after cleaning for {ticker} ({interval})")
                        continue

                    # Dopasuj nazwy kolumn do formatu yfinance
                    df.rename(columns={"open": "Open", "high": "High", "low": "Low", 
                                     "close": "Close", "adjclose": "Adj Close", "volume": "Volume"}, 
                            inplace=True)
                    
                    filename = f"{ticker}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = os.path.join("data/raw", filename)
                    df.to_csv(filepath, sep=";", decimal=".", index=False, encoding="utf-8-sig")
                    logging.info(f"Data downloaded for {ticker} ({interval}) to {filepath}")
                else:
                    logging.warning(f"No data for {ticker} ({interval})")
            except Exception as e:
                logging.error(f"Failed to download {ticker} ({interval}): {e}")
        
        # Opóźnienie między tickerami
        time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data with yahoo_fin")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA", help="Comma-separated list of tickers")
    parser.add_argument("--intervals", type=str, default="1d,1h,1wk", help="Comma-separated list of intervals")
    parser.add_argument("--days", type=int, default=730, help="Number of days to download for 1d, 1wk")
    parser.add_argument("--days_1h", type=int, default=60, help="Number of days to download for 1h")
    args = parser.parse_args()
    download_data(tickers=args.tickers.split(","), intervals=args.intervals.split(","), 
                  days=args.days, days_1h=args.days_1h)