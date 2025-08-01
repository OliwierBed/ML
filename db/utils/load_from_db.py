"""Utility for fetching market data from database or local snapshot."""

import os
import sqlite3
import pandas as pd


def load_data_from_db(ticker: str, interval: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return dataframe with candle data for *ticker* and *interval*.

    The function first attempts to read data from the Postgres database
    configured for the project (via :func:`db.session.get_db`).  When the
    database is unreachable—common when running locally without Docker—it
    gracefully falls back to a bundled SQLite database located at
    ``data-pipelines/feature_stores/data/database.db``.  Only selected
    *columns* are returned if specified.
    """

    try:
        # Attempt to load using the ORM (Postgres)
        from db.models import Candle
        from db.session import get_db

        db = next(get_db())
        query = (
            db.query(Candle)
            .filter(
                Candle.ticker == ticker,
                Candle.interval == interval,
                Candle.source == "raw",
            )
            .order_by(Candle.timestamp.asc())
        )
        df = pd.read_sql(query.statement, db.bind)
    except Exception:
        # Fallback: read from SQLite snapshot
        sqlite_path = os.path.join(
            "data-pipelines", "feature_stores", "data", "database.db"
        )
        if not os.path.exists(sqlite_path):
            raise
        conn = sqlite3.connect(sqlite_path)
        query = (
            "SELECT date, open, high, low, close, volume "
            "FROM stock_data WHERE ticker=? AND interval=? ORDER BY date ASC"
        )
        df = pd.read_sql(query, conn, params=(ticker, interval))
        conn.close()

    if columns is not None:
        columns_lower = [c.lower() for c in columns]
        # Always keep date/timestamp column if present to preserve time information
        for time_col in ("date", "timestamp"):
            if time_col in df.columns and time_col not in columns_lower:
                columns_lower.append(time_col)
        df = df[[col for col in df.columns if col.lower() in columns_lower]]

    return df

