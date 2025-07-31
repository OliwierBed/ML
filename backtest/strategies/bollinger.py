import pandas as pd

class BollingerBandsStrategy:
    def __init__(self, data, window=20, num_std=2):
        self.data = data
        self.window = window
        self.num_std = num_std

    def generate_signals(self):
        df = self.data.copy()
        df["rolling_mean"] = df["close"].rolling(window=self.window).mean()
        df["rolling_std"] = df["close"].rolling(window=self.window).std()
        df["upper_band"] = df["rolling_mean"] + self.num_std * df["rolling_std"]
        df["lower_band"] = df["rolling_mean"] - self.num_std * df["rolling_std"]
        df["signal"] = 0
        df.loc[df["close"] < df["lower_band"], "signal"] = 1
        df.loc[df["close"] > df["upper_band"], "signal"] = -1
        return df[["signal"]]