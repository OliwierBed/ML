# backtest/runner.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from backtest.rules import MACDCrossoverStrategy
from backtest.portfolio import Portfolio
from backtest.evaluation import evaluate
from backtest.utils import annualization_factor_from_interval

CONFIG_PATH = "config/config.yaml"
PROCESSED_DIR = "data-pipelines/feature_stores/data/processed"

def load_config():
    cfg = OmegaConf.load(CONFIG_PATH)
    if "backtest" not in cfg:
        raise KeyError("Brakuje sekcji 'backtest' w config/config.yaml")
    return cfg

def find_matching_file(ticker: str, interval: str) -> str:
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".csv") and ticker in filename and interval in filename:
            return os.path.join(PROCESSED_DIR, filename)
    raise FileNotFoundError(
        f"❌ Nie znaleziono pliku CSV dla {ticker} ({interval}) w {PROCESSED_DIR}"
    )

def run_backtest():
    # 1) Konfiguracja
    config = load_config()
    ticker = str(config.backtest.ticker)
    interval = str(config.backtest.interval)
    initial_cash = float(config.backtest.get("initial_cash", 10_000.0))

    print(f"\n🔍 Uruchamiam backtest dla {ticker} ({interval})...")

    # 2) Plik wejściowy
    path = find_matching_file(ticker, interval)
    filename = os.path.basename(path)
    af = annualization_factor_from_interval(interval)

    # 3) Dane
    df = pd.read_csv(path, sep=";")
    df.columns = [c.lower() for c in df.columns]

    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        raise ValueError("Brak kolumny z datą w pliku wejściowym.")

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # 4) Strategia -> wstrzykujemy sygnał do df
    strat = MACDCrossoverStrategy(df)
    strat.generate_signals()
    df["signal"] = strat.signals["signal"]

    # 5) Backtest
    portfolio = Portfolio(df, initial_cash=initial_cash)  # <--- TYLKO df + initial_cash
    results = portfolio.run()

    # 6) Metryki
    metrics = evaluate(results, af)

    print("\n📊 Statystyki strategii:")
    for k, v in metrics.items():
        if k in ("max_drawdown", "cagr"):
            print(f"{k:15s}: {v:.2%}")
        else:
            print(f"{k:15s}: {v:.4f}")

    # 7) Wykres
    ax = results[["equity_curve", "bh_equity_curve"]].plot(
        figsize=(10, 5),
        title=f"Equity curve – {filename} (strategia vs. buy&hold)",
        grid=True,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    plt.tight_layout()
    plt.show()

    # 8) Zapis
    os.makedirs("backtest/results", exist_ok=True)
    out_csv = f"backtest/results/{filename.replace('.csv', '')}_backtest.csv"
    results.to_csv(out_csv)
    print(f"\n💾 Zapisano wyniki do: {out_csv}")

    return results, metrics

if __name__ == "__main__":
    run_backtest()
