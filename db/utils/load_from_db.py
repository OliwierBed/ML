import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import yaml

# 📅 Wczytaj konfigurację
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 📂 Połączenie z bazą danych
db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=config["database"]["user"],
    password=config["database"]["password"],
    host=config["database"]["host"],
    port=config["database"]["port"],
    database=config["database"]["name"],
)
engine = create_engine(db_url)

# 🔍 Funkcja do wczytywania danych z bazy
def load_data_from_db(ticker: str, interval: str, source: str = "processed", strategy: str = None):
    query = """
        SELECT * FROM candles
        WHERE ticker = %(ticker)s
        AND interval = %(interval)s
        AND source = 'processed'
        AND (%(strategy)s IS NULL OR strategy = %(strategy)s)
        ORDER BY timestamp
    """
    params = {"ticker": ticker, "interval": interval, "strategy": strategy}
    df = pd.read_sql(query, engine, params=params)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    return df

# 🔎 Test lokalny
if __name__ == "__main__":
    df = load_from_db("AAPL", "1d", strategy="macd")
    print(df.head())
