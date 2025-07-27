# backtest/utils.py
import os
import re
import pandas as pd


INTERVAL_TO_AF = {
    "1h": 252 * 6.5,   # uproszczenie; dla krypto możesz dać 24*365
    "1d": 252,
    "1wk": 52,
}


def infer_interval_from_filename(filename: str) -> str:
    """
    Próbujemy wyciągnąć interwał z nazwy pliku, np. AAPL_1h_2025....csv -> 1h
    """
    m = re.search(r"_(1h|1d|1wk)_", filename)
    return m.group(1) if m else "1d"


def annualization_factor_from_interval(interval: str) -> float:
    return INTERVAL_TO_AF.get(interval, 252)


def load_processed_csv(path: str) -> pd.DataFrame:
    """
    Wczytuje CSV z kolumną Date; usuwa strefę czasową (tz) by matplotlib nie wariował
    """
    df = pd.read_csv(path, sep=";", parse_dates=["date"])
    # jeśli już ma tz, rzućmy na naive
    if hasattr(df["date"].dt, "tz_localize"):
        try:
            df["date"] = df["date"].dt.tz_localize(None)
        except TypeError:
            # już jest naive
            pass
    df.set_index("date", inplace=True)
    return df
