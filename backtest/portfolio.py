import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

class BacktestEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data.columns = [col.lower() for col in self.data.columns]

    def run(self):
        self._prepare_positions()
        self._calculate_returns()
        self._calculate_equity_curve()
        self._calculate_metrics()
        return self.results

    def _prepare_positions(self):
        self.data["position"] = self.data["signal"].replace(0, pd.NA).ffill().fillna(0)

    def _calculate_returns(self):
        self.data["strategy_returns"] = self.data["Close"].pct_change().fillna(0) * self.data["position"]
        self.data["equity_curve"] = (1 + self.data["strategy_returns"]).cumprod() * self.initial_cash

        # Buy & Hold
        self.data["bh_returns"] = self.data["Close"].pct_change().fillna(0)
        self.data["bh_equity_curve"] = (1 + self.data["bh_returns"]).cumprod() * self.initial_cash

    def _calculate_metrics(self):
        returns = self.data["strategy_returns"]
        bh_returns = self.data["bh_returns"]

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() != 0 else 0
        max_drawdown = (self.data["equity_curve"] / self.data["equity_curve"].cummax() - 1).min()

        win_rate = (self.data["strategy_returns"] > 0).sum() / len(self.data)

        self.results = {
            "sharpe": round(sharpe, 6),
            "sortino": round(sortino, 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 6),
            "cagr": np.nan,
            "mar": np.nan,
            "final_equity": self.data["equity_curve"].iloc[-1],
        }

    def plot(self, filename="strategia"):
        self.data[["equity_curve", "bh_equity_curve"]].plot(figsize=(10, 6))
        plt.title(f"Equity curve – {filename} (strategia vs. buy&hold)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio value ($)")
        plt.legend()
        plt.tight_layout()
        plt.show()
