# backtest/portfolio.py
import pandas as pd
import numpy as np

from backtest.evaluate import evaluate_backtest

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_cash: float, annualization_factor: int):
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.af = annualization_factor
        if "close" not in self.data.columns:
            raise ValueError("Brak kolumny 'close' w danych!")

    def run(self, signal_col: str = "signal"):
        if signal_col not in self.data.columns:
            raise ValueError(f"Brak kolumny '{signal_col}' w danych!")

        # pozycja (1, -1, 0) – wypełniamy 0 -> ffillem, żeby utrzymać pozycję
        position = self.data[signal_col].replace(0, np.nan).ffill().fillna(0)
        ret = self.data["close"].pct_change(fill_method=None).fillna(0)

        # strategia używa pozycji z poprzedniej świecy
        strat_ret = ret * position.shift(1).fillna(0)

        equity_curve = (1 + strat_ret).cumprod() * self.initial_cash
        bh_equity_curve = (1 + ret).cumprod() * self.initial_cash

        self.data["position"] = position
        self.data["strategy_returns"] = strat_ret
        self.data["equity_curve"] = equity_curve
        self.data["bh_equity_curve"] = bh_equity_curve

        metrics = evaluate_backtest(strat_ret, equity_curve, self.af)
        return self.data, metrics
