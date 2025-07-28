# backtest/rules.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from backtest.rules_lstm import LSTMStrategy

# ============== Base ==============

@dataclass
class BaseStrategy:
    df: pd.DataFrame

    def _need_cols(self, cols):
        for c in cols:
            if c not in self.df.columns:
                raise KeyError(f"Brak kolumny '{c}' potrzebnej w strategii {self.__class__.__name__}")

    def _ffill_positions(self, signal: pd.Series) -> pd.Series:
        return signal.replace(0, np.nan).ffill().fillna(0)

    def generate_signals(self) -> pd.DataFrame:
        raise NotImplementedError


# ============== MACD ==============

class MACDStrategy(BaseStrategy):
    """
    Jeśli w df nie ma macd / macd_signal – liczymy je on-the-fly z EMA(12) i EMA(26).
    Sygnał = 1 gdy macd przecina w górę macd_signal, -1 gdy w dół, inaczej 0 (ffill).
    """

    def _compute_macd(self):
        close = self.df["close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        return macd, macd_signal

    def generate_signals(self) -> pd.DataFrame:
        if "macd" not in self.df.columns or "macd_signal" not in self.df.columns:
            macd, macd_signal = self._compute_macd()
        else:
            macd = self.df["macd"]
            macd_signal = self.df["macd_signal"]

        cross_up = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
        cross_dn = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))

        signal = pd.Series(0, index=self.df.index, dtype=float)
        signal[cross_up] = 1.0
        signal[cross_dn] = -1.0
        signal = self._ffill_positions(signal)

        return pd.DataFrame({"signal": signal})


# ============== RSI ==============

class RSIStrategy(BaseStrategy):
    """
    Jeśli nie ma rsi_14 – liczymy RSI(14) samodzielnie.
    Kupno gdy RSI < 30, sprzedaż (short) gdy RSI > 70, inaczej 0 (ffill).
    """

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=period - 1, adjust=False).mean()
        ma_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self) -> pd.DataFrame:
        if "rsi_14" in self.df.columns:
            rsi = self.df["rsi_14"]
        else:
            rsi = self._rsi(self.df["close"], period=14)

        signal = pd.Series(0, index=self.df.index, dtype=float)
        signal[rsi < 30] = 1.0
        signal[rsi > 70] = -1.0
        signal = self._ffill_positions(signal)

        return pd.DataFrame({"signal": signal})


# ============== SMA (price vs SMA20) ==============

class SMAStrategy(BaseStrategy):
    """
    Jeśli nie ma sma_20 – liczymy ją samodzielnie.
    Pozycja 1 gdy close > SMA20, -1 gdy close < SMA20.
    """

    def generate_signals(self) -> pd.DataFrame:
        if "sma_20" in self.df.columns:
            sma = self.df["sma_20"].bfill()
        else:
            sma = self.df["close"].rolling(20).mean().bfill()

        close = self.df["close"]
        signal = pd.Series(0, index=self.df.index, dtype=float)
        signal[close > sma] = 1.0
        signal[close < sma] = -1.0
        signal = self._ffill_positions(signal)

        return pd.DataFrame({"signal": signal})


# ============== EMA cross (12 / 26) ==============

class EMACrossStrategy(BaseStrategy):
    """
    Sygnał 1 gdy EMA12 > EMA26, -1 gdy EMA12 < EMA26.
    """

    def generate_signals(self) -> pd.DataFrame:
        close = self.df["close"]
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()

        signal = pd.Series(0, index=self.df.index, dtype=float)
        signal[ema_fast > ema_slow] = 1.0
        signal[ema_fast < ema_slow] = -1.0
        signal = self._ffill_positions(signal)
        return pd.DataFrame({"signal": signal})


# ============== Bollinger (mean reversion) ==============

class BollingerBreakoutStrategy(BaseStrategy):
    """
    Liczymy BB(20, 2). Gdy close < dolnego pasma => +1 (kup), gdy > górnego => -1 (sprzedaj).
    Prosty mean reversion / breakout mix.
    """

    def generate_signals(self) -> pd.DataFrame:
        close = self.df["close"]
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std

        signal = pd.Series(0, index=self.df.index, dtype=float)
        signal[close < lower] = 1.0
        signal[close > upper] = -1.0
        signal = self._ffill_positions(signal)
        return pd.DataFrame({"signal": signal})


# ============== Fabryka strategii ==============

def get_strategy(name: str):
    name = name.lower()
    mapping = {
        "macd": MACDStrategy,
        "rsi": RSIStrategy,
        "sma": SMAStrategy,
        "ema": EMACrossStrategy,
        "bollinger": BollingerBreakoutStrategy,
    }

    if name == "lstm":
        return lambda df, **kwargs: LSTMStrategy(df, kwargs.get("ticker"), kwargs.get("interval"))
    if name not in mapping:
        raise ValueError(f"Strategia '{name}' nie jest obsługiwana. Dostępne: {list(mapping.keys())}")
    return mapping[name]



    
