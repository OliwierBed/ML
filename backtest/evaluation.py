# backtest/evaluation.py
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, af: float, risk_free: float = 0.0) -> float:
    r = returns.dropna()
    if r.std() == 0:
        return 0.0
    return ((r.mean() - risk_free / af) / r.std()) * np.sqrt(af)


def sortino_ratio(returns: pd.Series, af: float, risk_free: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if downside.std() == 0:
        return np.inf
    return ((r.mean() - risk_free / af) / downside.std()) * np.sqrt(af)


def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return dd.min()


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return (r > 0).sum() / len(r)


def mar_ratio(cagr_val: float, max_dd: float) -> float:
    if max_dd == 0:
        return np.inf
    return cagr_val / abs(max_dd)


def evaluate(results: pd.DataFrame, af: float) -> dict:
    r = results["strategy_returns"]
    equity = results["equity_curve"]

    sr = sharpe_ratio(r, af)
    sor = sortino_ratio(r, af)
    mdd = max_drawdown(equity)
    wr = win_rate(r)
    c = cagr(equity, af)
    mar = mar_ratio(c, mdd)

    return {
        "sharpe": sr,
        "sortino": sor,
        "max_drawdown": mdd,
        "win_rate": wr,
        "cagr": c,
        "mar": mar,
        "final_equity": equity.iloc[-1],
    }
