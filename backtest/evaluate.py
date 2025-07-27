# backtest/evaluate.py
import numpy as np
import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2:
        return 0.0
    start_val = equity.iloc[0]
    end_val = equity.iloc[-1]
    n_periods = len(equity)
    years = n_periods / periods_per_year
    if start_val <= 0 or years <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / years) - 1


def evaluate_backtest(strategy_returns: pd.Series,
                      equity_curve: pd.Series,
                      annualization_factor: int) -> dict:
    ret = strategy_returns.dropna()

    if ret.std() != 0:
        sharpe = ret.mean() / ret.std() * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0

    downside = ret[ret < 0].std()
    if downside != 0 and not np.isnan(downside):
        sortino = ret.mean() / downside * np.sqrt(annualization_factor)
    else:
        sortino = 0.0

    mdd = max_drawdown(equity_curve)
    win_rate = float((ret > 0).mean())
    the_cagr = cagr(equity_curve, annualization_factor)
    mar = the_cagr / abs(mdd) if mdd != 0 else float("inf")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "cagr": the_cagr,
        "mar": mar,
        "final_equity": float(equity_curve.iloc[-1]),
    }
