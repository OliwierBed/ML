import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Candle
from datetime import datetime
import yaml

# Wczytaj konfigurację DB
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_cfg = config["database"]
DATABASE_URL = f"postgresql://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def load_candles(ticker: str, interval: str, source: str, strategy: str = None) -> pd.DataFrame:
    query = session.query(Candle).filter_by(ticker=ticker, interval=interval, source=source)
    if strategy:
        query = query.filter_by(strategy=strategy)
    else:
        query = query.filter(Candle.strategy == None)
    df = pd.read_sql(query.statement, session.bind)
    df = df.sort_values("date").reset_index(drop=True)
    return df
