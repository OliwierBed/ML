# backtest/runner.py
import os
import pandas as pd
import matplotlib.pyplot as plt

from backtest.rules import MACDCrossoverStrategy
from backtest.portfolio import Portfolio
from backtest.utils import (
    load_processed_csv,
    infer_interval_from_filename,
    annualization_factor_from_interval,
)
from backtest.evaluation import evaluate


def run_backtest(
    path: str = "data-pipelines/feature_stores/data/processed/AAPL_1h_20250724_111413_indicators.csv",
    initial_cash: float = 10_000.0,
):
    # 1) Dane
    filename = os.path.basename(path)
    interval = infer_interval_from_filename(filename)
    af = annualization_factor_from_interval(interval)

    df = load_processed_csv(path)

    # 2) Strategia
    strat = MACDCrossoverStrategy(df)
    strat.generate_signals()

    # 3) Symulacja portfela (z buy&hold baseline)
    portfolio = Portfolio(df, strat.signals, initial_cash=initial_cash)
    results = portfolio.run()

    # 4) Metryki
    metrics = evaluate(results, af)

    print("\n📊 Statystyki strategii:")
    for k, v in metrics.items():
        if k in ("max_drawdown", "cagr"):
            print(f"{k:15s}: {v:.2%}")
        else:
            print(f"{k:15s}: {v:.4f}")

    # 5) Wykres equity curve vs buy&hold
    ax = results[["equity_curve", "bh_equity_curve"]].plot(
        figsize=(10, 5), title=f"Equity curve – {filename} (strategia vs. buy&hold)", grid=True
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    plt.tight_layout()
    plt.show()

    # 6) Zapis wyników
    os.makedirs("backtest/results", exist_ok=True)
    out_csv = f"backtest/results/{filename.replace('.csv', '')}_backtest.csv"
    results.to_csv(out_csv)
    print(f"\n💾 Zapisano wyniki do: {out_csv}")

    return results, metrics


if __name__ == "__main__":
    run_backtest()
