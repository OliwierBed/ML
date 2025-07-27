import pandas as pd

class BaseStrategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_signals(self) -> pd.DataFrame:
        raise NotImplementedError


class MACDStrategy(BaseStrategy):
    def generate_signals(self):
        self.df["signal"] = 0
        self.df.loc[self.df["macd"] > self.df["macd_signal"], "signal"] = 1
        self.df.loc[self.df["macd"] < self.df["macd_signal"], "signal"] = -1
        self.df["signal"] = self.df["signal"].replace(0, pd.NA).ffill().fillna(0)
        return self.df


class RSIStrategy(BaseStrategy):
    def generate_signals(self):
        self.df["signal"] = 0
        self.df.loc[self.df["rsi_14"] < 30, "signal"] = 1
        self.df.loc[self.df["rsi_14"] > 70, "signal"] = -1
        self.df["signal"] = self.df["signal"].replace(0, pd.NA).ffill().fillna(0)
        return self.df


class SMAStrategy(BaseStrategy):
    def generate_signals(self):
        self.df["signal"] = 0
        sma = self.df["sma_20"].fillna(method="bfill")
        self.df.loc[self.df["close"] > sma, "signal"] = 1
        self.df.loc[self.df["close"] < sma, "signal"] = -1
        self.df["signal"] = self.df["signal"].replace(0, pd.NA).ffill().fillna(0)
        return self.df


class EMACrossStrategy(BaseStrategy):
    def generate_signals(self):
        self.df["signal"] = 0
        self.df.loc[self.df["ema_10"] > self.df["ema_50"], "signal"] = 1
        self.df.loc[self.df["ema_10"] < self.df["ema_50"], "signal"] = -1
        self.df["signal"] = self.df["signal"].replace(0, pd.NA).ffill().fillna(0)
        return self.df


class BollingerBreakoutStrategy(BaseStrategy):
    def generate_signals(self):
        self.df["signal"] = 0
        self.df.loc[self.df["close"] > self.df["bollinger_upper"], "signal"] = 1
        self.df.loc[self.df["close"] < self.df["bollinger_lower"], "signal"] = -1
        self.df["signal"] = self.df["signal"].replace(0, pd.NA).ffill().fillna(0)
        return self.df


def get_strategy(name: str):
    name = name.lower()
    if name == "macd":
        return MACDStrategy
    elif name == "rsi":
        return RSIStrategy
    elif name == "sma":
        return SMAStrategy
    elif name == "ema":
        return EMACrossStrategy
    elif name == "bollinger":
        return BollingerBreakoutStrategy
    else:
        raise ValueError(f"Strategia '{name}' nie jest obsługiwana.")
