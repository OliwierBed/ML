# backtest/strategies/ensemble.py
import pandas as pd

class EnsembleStrategy:
    def __init__(self, data):
        self.data = data

    def generate_signals(self):
        df = self.data.copy()
        signal_cols = [col for col in df.columns if col.startswith("signal_")]
        if not signal_cols:
            raise ValueError("Brak kolumn 'signal_*' do stworzenia ensemble.")
        df["signal"] = df[signal_cols].sum(axis=1).clip(-1, 1)
        return df[["signal"]]
