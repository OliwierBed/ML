from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.signals = pd.Series(0, index=self.df.index)

    @abstractmethod
    def generate_signals(self):
        pass

    def run(self):
        self.generate_signals()
        return self.signals
