import os
import pandas as pd
import numpy as np

ENSEMBLE_DIR = "data-pipelines/feature_stores/data/processed/ensemble"
RESULTS_DIR = "data-pipelines/feature_stores/data/results"
OUT_CSV = os.path.join(RESULTS_DIR, "ensemble_batch_results.csv")

def annualization_factor(interval):
    return {
        "1h": 252 * 6.5,  # zakładając 6,5h sesji (USA)
        "1d": 252,
        "1wk": 52,
    }.get(interval, 252)

def compute_metrics(df, fname, initial_cash=100_000):
    # Zakładamy: kolumny ['date', 'signal', 'close']
    df = df.dropna(subset=['close'])
    df = df.sort_values('date')
    equity = []
    cash = initial_cash
    position = 0
    entry_price = None
    for i, row in df.iterrows():
        price = row['close']
        signal = row['signal']
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 1:
            cash *= price / entry_price
            position = 0
            entry_price = None
        eq_val = cash if position == 0 or entry_price is None else cash * price / entry_price
        equity.append(eq_val)
    if position == 1 and entry_price is not None:
        cash *= price / entry_price
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    if len(returns) == 0:
        returns = np.array([0])
    # Metryki:
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = np.mean(returns) / (np.std(downside) + 1e-9) * np.sqrt(252) if len(downside) > 0 else 0
    rolling_max = np.maximum.accumulate(equity)
    drawdown = equity / rolling_max - 1.0
    max_drawdown = np.min(drawdown)
    win_rate = np.mean(returns > 0)
    total_return = equity[-1] / initial_cash - 1
    years = (len(df) / annualization_factor(infer_interval_from_filename(fname)))
    cagr = (equity[-1] / initial_cash) ** (1/years) - 1 if years > 0 else total_return
    mar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "sharpe": round(float(sharpe), 6),
        "sortino": round(float(sortino), 6),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "cagr": float(cagr),
        "mar": float(mar) if not np.isnan(mar) else None,
        "final_equity": float(equity[-1]),
    }

def infer_ticker_interval(fname):
    # Przykład: AAPL_1d_ensemble_full.csv
    base = os.path.basename(fname)
    parts = base.split("_")
    ticker = parts[0]
    interval = parts[1]
    return ticker, interval

def infer_interval_from_filename(fname):
    # Przykład: AAPL_1d_ensemble_full.csv
    base = os.path.basename(fname)
    parts = base.split("_")
    return parts[1] if len(parts) > 1 else "1d"

def main():
    rows = []
    for fname in os.listdir(ENSEMBLE_DIR):
        if fname.endswith("_ensemble_full.csv"):
            fpath = os.path.join(ENSEMBLE_DIR, fname)
            try:
                df = pd.read_csv(fpath, sep=";")
            except Exception:
                continue
            if "close" not in df.columns or "signal" not in df.columns:
                continue
            metrics = compute_metrics(df, fname)
            ticker, interval = infer_ticker_interval(fname)
            row = {
                **metrics,
                "ticker": ticker,
                "interval": interval,
                "strategy": "ensemble",
                "filename": fname,
            }
            rows.append(row)
            print(f"✔ Przeliczono metryki dla {fname}: {metrics}")

    if not rows:
        print("Nie znaleziono żadnych plików ensemble do przetworzenia.")
        return

    dfout = pd.DataFrame(rows)
    dfout.to_csv(OUT_CSV, sep=";", index=False)
    print(f"✅ Zapisano wyniki ensemble do: {OUT_CSV}")

if __name__ == "__main__":
    main()
