# backtest/utils.py
import os
import pandas as pd

INTERVALS = ["1h", "1d", "1wk"]

def load_processed_csv_lower(path: str) -> pd.DataFrame:
    """
    Wczytuje csv, zamienia nazwy kolumn na lowercase, wykrywa kolumnę daty.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = [c.lower() for c in df.columns]

    # Szukamy kolumny z datą
    date_col = None
    for cand in ("date", "datetime", "ts"):
        if cand in df.columns:
            date_col = cand
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

    return df


def infer_interval_from_filename(filename: str) -> str:
    for itv in INTERVALS:
        # najpewniej w formacie _1h_ / _1d_ / _1wk_
        if f"_{itv}_" in filename:
            return itv
    return "1d"


def annualization_factor_from_interval(interval: str) -> int:
    return {
        "1h": 24 * 252,
        "1d": 252,
        "1wk": 52,
    }.get(interval, 252)
