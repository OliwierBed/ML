import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime
import argparse
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_indicators(
    ticker: str, interval: str = "1d", input_dir: str = None, output_dir: str = None
) -> None:
    config = load_config()
    # Pobierz ścieżki z configa jeśli nie podano
    if input_dir is None:
        input_dir = config["data"].get("raw_data_dir", "data-pipelines/feature_stores/data/raw")
    if output_dir is None:
        output_dir = config["data"].get("processed_data_dir", "data-pipelines/feature_stores/data/processed")
    # Znajdź pliki dla danego tickera i interwału
    files = [f for f in os.listdir(input_dir) if f.startswith(f"{ticker}_{interval}_") and f.endswith(".csv")]
    if not files:
        logging.error(f"No raw data file found for {ticker} ({interval})")
        return
    
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))
    file_path = os.path.join(input_dir, latest_file)
    # Wczytaj dane
    try:
        df = pd.read_csv(file_path, sep=";", decimal=".", parse_dates=["Date"], encoding="utf-8-sig")
        if "Date" not in df.columns:
            df = pd.read_csv(file_path, sep=";", decimal=".", encoding="utf-8-sig")
            df["Date"] = pd.to_datetime(df.index, errors="coerce")
            df = df.reset_index(drop=True)
        logging.info(f"Wczytano dane dla {ticker} ({interval}) z pliku {file_path}, liczba wierszy: {len(df)}")
        logging.info(f"Kolumny danych: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Błąd wczytywania danych dla {ticker} ({interval}): {str(e)}")
        return
    
    # Sprawdź wymagane kolumny
    required_cols = ["Close", "High", "Low", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Brak wymaganych kolumn dla {ticker} ({interval}): {missing_cols}")
        return
    
    # Konwertuj kolumny na numeryczne
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().all():
            logging.error(f"Wszystkie wartości w kolumnie {col} są NaN dla {ticker} ({interval})")
            return
        if df[col].isna().any():
            logging.info(f"Wypełnianie NaN w kolumnie {col} dla {ticker} ({interval})")
            df[col] = df[col].ffill().bfill()
    
    # Przemianuj kolumny dla spójności
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Adj_Close"})
    
    # Oblicz wskaźniki
    try:
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
        
        # Bollinger Bands
        bbands = ta.bbands(df["Close"], length=20, std=2)
        if bbands is not None and isinstance(bbands, pd.DataFrame):
            logging.info(f"Kolumny Bollinger Bands: {bbands.columns.tolist()}")
            df["BB_Low"] = bbands[bbands.columns[0]]  # BBL
            df["BB_Mid"] = bbands[bbands.columns[1]]  # BBM
            df["BB_High"] = bbands[bbands.columns[2]] # BBU
        else:
            logging.warning(f"Nie udało się obliczyć Bollinger Bands dla {ticker} ({interval})")
            df["BB_Low"] = 0.0
            df["BB_Mid"] = 0.0
            df["BB_High"] = 0.0
        
        # Stochastic Oscillator
        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
        if stoch is not None and isinstance(stoch, pd.DataFrame):
            logging.info(f"Kolumny Stochastic: {stoch.columns.tolist()}")
            df["Stoch_K"] = stoch[stoch.columns[0]]  # STOCHk
            df["Stoch_D"] = stoch[stoch.columns[1]]  # STOCHd
        else:
            logging.warning(f"Nie udało się obliczyć Stochastic Oscillator dla {ticker} ({interval})")
            df["Stoch_K"] = 0.0
            df["Stoch_D"] = 0.0
        
        # MACD
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None and isinstance(macd, pd.DataFrame):
            logging.info(f"Kolumny MACD: {macd.columns.tolist()}")
            df["MACD"] = macd[macd.columns[0]]  # MACD
            df["MACD_Signal"] = macd[macd.columns[1]]  # MACDs
            df["MACD_Hist"] = macd[macd.columns[2]]  # MACDh
        else:
            logging.warning(f"Nie udało się obliczyć MACD dla {ticker} ({interval})")
            df["MACD"] = 0.0
            df["MACD_Signal"] = 0.0
            df["MACD_Hist"] = 0.0
        
        # Normalizacja Close
        df["Close_Norm"] = (df["Close"] - df["Close"].min()) / (df["Close"].max() - df["Close"].min())
        
        # Usuń wiersze z NaN w kluczowych wskaźnikach
        indicator_columns = [
            "Close", "Adj_Close", "Close_Norm", "SMA_20", "RSI_14", "ATR_14", "OBV", 
            "BB_Low", "BB_Mid", "BB_High", "Stoch_K", "Stoch_D", 
            "MACD", "MACD_Signal", "MACD_Hist"
        ]
        df = df.dropna(subset=indicator_columns)
        
        # Zapisz dane
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{ticker}_{interval}_{timestamp}_indicators.csv")
        df.to_csv(output_path, sep=";", decimal=".", encoding="utf-8-sig", index=False)
        logging.info(f"Dane z wskaźnikami zapisane w {output_path}")
        
    except Exception as e:
        logging.error(f"Błąd podczas obliczania wskaźników dla {ticker} ({interval}): {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process technical indicators")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers")
    parser.add_argument("--intervals", type=str, default=None, help="Comma-separated list of intervals")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory for raw data")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for processed data")
    args = parser.parse_args()

    config = load_config()
    tickers = args.tickers.split(",") if args.tickers else config["data"].get("tickers", ["AAPL", "MSFT", "TSLA"])
    intervals = args.intervals.split(",") if args.intervals else config["data"].get("intervals", ["1d", "1h", "1wk"])

    for ticker in tickers:
        for interval in intervals:
            interval = interval.strip()
            if not interval: continue
            process_indicators(
                ticker, 
                interval, 
                input_dir=args.input_dir, 
                output_dir=args.output_dir
            )
