import yaml
from sqlalchemy import create_engine, text

# Wczytaj konfigurację z config.yaml
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_cfg = config["database"]

db_url = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['name']}"

engine = create_engine(db_url)

with engine.begin() as conn:
    # Ustaw tymczasowo wartość timestamp, jeśli NULL
    conn.execute(text("""
        UPDATE candles
        SET timestamp = NOW()
        WHERE timestamp IS NULL
    """))

print("✅ Naprawiono rekordy z NULL w kolumnie 'timestamp'.")
