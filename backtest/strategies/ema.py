import pandas as pd

class EMAStrategy:
    def __init__(self, data, span=20):
        self.data = data
        self.span = span

    def generate_signals(self):
        df = self.data.copy()
        df["ema"] = df["close"].ewm(span=self.span, adjust=False).mean()
        df["signal"] = 0
        df.loc[df["close"] > df["ema"], "signal"] = 1
        df.loc[df["close"] < df["ema"], "signal"] = -1
        return df[["signal"]]
