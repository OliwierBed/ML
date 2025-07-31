from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys
import os

# Dodaj ścieżkę do katalogu głównego projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.models import Base  # <-- import modelu
from db.session import get_db_url

config = context.config
fileConfig(config.config_file_name)

# Ustaw URL dynamicznie z YAML
config.set_main_option("sqlalchemy.url", get_db_url())

target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
