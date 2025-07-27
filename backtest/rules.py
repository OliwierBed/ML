import pandas as pd
from backtest.base import BaseStrategy

class MACDCrossoverStrategy(BaseStrategy):
    def generate_signals(self):
        self.df['macd_hist'] = self.df['MACD_Hist']

        self.signals = pd.Series(0, index=self.df.index)

        for i in range(1, len(self.df)):
            if self.df['macd_hist'].iloc[i] > 0 and self.df['macd_hist'].iloc[i - 1] <= 0:
                self.signals.iloc[i] = 1  # Kup
            elif self.df['macd_hist'].iloc[i] < 0 and self.df['macd_hist'].iloc[i - 1] >= 0:
                self.signals.iloc[i] = -1  # Sprzedaj
