from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.get("/stocks/{ticker}/{interval}")
async def get_stock_data(ticker: str, interval: str):
    try:
        conn = sqlite3.connect("data/database.db")
        query = "SELECT * FROM stock_data WHERE ticker = ? AND interval = ?"
        df = pd.read_sql_query(query, conn, params=(ticker, interval))
        conn.close()
        
        if df.empty:
            logging.warning(f"No stock data found for ticker {ticker} and interval {interval}")
            raise HTTPException(status_code=404, detail=f"No stock data found for {ticker} ({interval})")
        
        logging.info(f"Fetched stock data for {ticker} ({interval}), rows: {len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker} ({interval}): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indicators/{ticker}/{interval}")
async def get_indicators(ticker: str, interval: str):
    try:
        conn = sqlite3.connect("data/database.db")
        query = "SELECT * FROM indicators WHERE ticker = ? AND interval = ?"
        df = pd.read_sql_query(query, conn, params=(ticker, interval))
        conn.close()
        
        if df.empty:
            logging.warning(f"No indicators found for ticker {ticker} and interval {interval}")
            raise HTTPException(status_code=404, detail=f"No indicators found for {ticker} ({interval})")
        
        logging.info(f"Fetched indicators for {ticker} ({interval}), rows: {len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Error fetching indicators for {ticker} ({interval}): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indicators/latest/{ticker}/{interval}")
async def get_latest_indicators(ticker: str, interval: str):
    try:
        conn = sqlite3.connect("data/database.db")
        query = "SELECT * FROM indicators WHERE ticker = ? AND interval = ? ORDER BY date DESC LIMIT 1"
        df = pd.read_sql_query(query, conn, params=(ticker, interval))
        conn.close()
        
        if df.empty:
            logging.warning(f"No latest indicators found for {ticker} ({interval})")
            raise HTTPException(status_code=404, detail=f"No latest indicators found for {ticker} ({interval})")
        
        logging.info(f"Fetched latest indicators for {ticker} ({interval})")
        return df.to_dict(orient="records")[0]
    except Exception as e:
        logging.error(f"Error fetching latest indicators for {ticker} ({interval}): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))