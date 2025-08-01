from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import yaml

from db.models import Base

def get_db_url():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        db_conf = config["database"]
        return (
            f"postgresql://{db_conf['user']}:{db_conf['password']}"
            f"@{db_conf['host']}:{db_conf['port']}/{db_conf['name']}"
        )

DATABASE_URL = get_db_url()

engine = create_engine(DATABASE_URL)

# Ensure required tables exist so API calls don't fail with "relation does not exist".
# If the database is unreachable we allow the import to continue so callers can
# handle connection errors gracefully later on.
try:
    Base.metadata.create_all(bind=engine)
except Exception:
    pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
