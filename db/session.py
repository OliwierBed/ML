from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from configparser import ConfigParser
import yaml

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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
