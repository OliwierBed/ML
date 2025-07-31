# backtest/strategies/macd.py
import pandas as pd

class MACDCrossoverStrategy:
    def __init__(self, data):
        self.data = data

    def generate_signals(self):
        df = self.data.copy()
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["signal"] = 0
        df.loc[df["macd"] > df["signal_line"], "signal"] = 1
        df.loc[df["macd"] < df["signal_line"], "signal"] = -1
        return df[["signal"]]
