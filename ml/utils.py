import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_close_series(df: pd.DataFrame, window: int = 10):
    """Apply rolling mean and MinMax scaling to close column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe expected to contain ``close`` column.
    window : int
        Rolling window size for moving average.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, MinMaxScaler]
        Tuple containing processed dataframe, scaled close series and fitted scaler.
    """
    data = df.copy()
    data.columns = data.columns.str.lower()
    if "close" not in data.columns:
        raise ValueError("Brak kolumny 'close' w danych.")

    data["close_ma"] = data["close"].rolling(window=window).mean()
    data = data.dropna(subset=["close_ma"]).reset_index(drop=True)
    if data.empty:
        raise ValueError(
            f"Za mało danych do obliczenia MA({window}). Dostępne: {len(df)}"
        )

    scaler = MinMaxScaler()
    data["close_scaled"] = scaler.fit_transform(data[["close_ma"]])

    return data, data["close_scaled"].values.astype("float32"), scaler
