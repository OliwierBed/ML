import pandas as pd

class SMAStrategy:
    def __init__(self, data, short_window=20, long_window=50):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        df = self.data.copy()
        df["sma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_window).mean()
        df["signal"] = 0
        df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
        df.loc[df["sma_short"] < df["sma_long"], "signal"] = -1
        return df[["signal"]]
