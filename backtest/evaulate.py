def evaluate_backtest(df, initial_cash):
    returns = df["equity_curve"].pct_change().dropna()
    final_equity = df["equity_curve"].iloc[-1]
    cagr = ((final_equity / initial_cash) ** (1 / (len(df) / 252))) - 1 if len(df) > 1 else 0
    max_drawdown = ((df["equity_curve"] / df["equity_curve"].cummax()) - 1).min()
    sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
    sortino = returns.mean() / returns[returns < 0].std() * (252 ** 0.5) if (returns < 0).std() > 0 else 0
    win_rate = (returns > 0).mean()
    mar = cagr / abs(max_drawdown) if max_drawdown != 0 else float("inf")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "cagr": cagr,
        "mar": mar,
        "final_equity": final_equity,
    }
