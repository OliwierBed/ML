"""Utility for fetching market data from the primary Postgres database."""

import pandas as pd


def load_data_from_db(
    ticker: str, interval: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Return dataframe with candle data for *ticker* and *interval*.

    All reads go through the configured Postgres database via
    :func:`db.session.get_db`.  If the database is unreachable the function
    will raise the underlying exception instead of falling back to local
    snapshots, ensuring a single source of truth for the application.
    """

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

    if columns is not None:
        columns_lower = [c.lower() for c in columns]
        # Always keep date/timestamp column if present to preserve time information
        for time_col in ("date", "timestamp"):
            if time_col in df.columns and time_col not in columns_lower:
                columns_lower.append(time_col)
        df = df[[col for col in df.columns if col.lower() in columns_lower]]

    return df

