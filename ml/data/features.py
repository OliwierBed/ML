import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def build_features(df: pd.DataFrame, feature_cols, target_col="close",
                   roll_mean=10, scale=True):
    df = df.copy()
    if roll_mean:
        df[target_col] = df[target_col].rolling(roll_mean).mean()
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values
    y = df[target_col].values

    scaler_x, scaler_y = None, None
    if scale:
        scaler_x = MinMaxScaler().fit(X)
        scaler_y = MinMaxScaler().fit(y.reshape(-1, 1))
        X = scaler_x.transform(X)
        y = scaler_y.transform(y.reshape(-1, 1)).flatten()

    return X, y, scaler_x, scaler_y
