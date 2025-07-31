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
def load_data_from_db(ticker: str, interval: str, columns: list[str] = None) -> pd.DataFrame:
    from db.models import Candle
    from db.session import get_db

    db = next(get_db())

    query = db.query(Candle).filter(
        Candle.ticker == ticker,
        Candle.interval == interval,
        Candle.source == "raw"
    ).order_by(Candle.timestamp.asc())

    df = pd.read_sql(query.statement, db.bind)

    # Domyślnie pobieraj wszystkie kolumny
    if columns is not None:
        columns_lower = [col.lower() for col in columns]
        df = df[[col for col in df.columns if col.lower() in columns_lower]]

    return df
