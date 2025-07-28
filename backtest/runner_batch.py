# backtest/runner_batch.py
import os
import pandas as pd
from omegaconf import OmegaConf

from backtest.utils import (
    load_processed_csv_lower,
    infer_interval_from_filename,
    annualization_factor_from_interval,
)
from backtest.rules import get_strategy
from backtest.portfolio import BacktestEngine


def main():
    config = OmegaConf.load("config/config.yaml")

    PROCESSED_DIR = config.paths.feature_stores_processed
    RESULTS_DIR = config.paths.results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    TICKERS = [t.upper() for t in config.data.tickers]
    INTERVALS = [i for i in config.data.intervals]
    STRATEGIES = [s.lower() for s in config.backtest.strategies]
    INITIAL_CASH = float(config.backtest.initial_cash)

    print(f"📁 Używam katalogu z danymi: {PROCESSED_DIR}")
    print(f"🧠 Strategie: {STRATEGIES}")

    all_results = []

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    if not files:
        print("⚠️ Brak plików CSV w katalogu processed.")
        return

    for filename in files:
        # filtr tickerów i interwałów
        if not any(t in filename for t in TICKERS):
            continue
        if not any(f"_{itv}_" in filename for itv in INTERVALS):
            continue

        ticker = next(t for t in TICKERS if t in filename)
        interval = infer_interval_from_filename(filename)
        af = annualization_factor_from_interval(interval)

        for strategy_name in STRATEGIES:
            print(f"\n🔁 Przetwarzam: {filename} [{strategy_name}]")
            full_path = os.path.join(PROCESSED_DIR, filename)
            try:
                df = load_processed_csv_lower(full_path)
                if "close" not in df.columns:
                    raise ValueError("Brak kolumny 'close' w danych!")

                StratCls = get_strategy(strategy_name)
                if strategy_name == "lstm":
                    strategy = StratCls(df, ticker=ticker, interval=interval)
                else:
                    strategy = StratCls(df)
                signals = strategy.generate_signals()

                # merge sygnałów do df
                if "signal" in df.columns:
                    df.drop(columns=["signal"], inplace=True)
                df["signal"] = signals["signal"]
                df_bt = df

                if df_bt["signal"].isna().all():
                    raise ValueError("Sygnał zawiera same NaN – sprawdź strategię.")

                engine = BacktestEngine(df_bt, initial_cash=INITIAL_CASH, annualization_factor=af)
                results_df, metrics = engine.run(signal_col="signal")

                # doklej meta
                metrics["ticker"] = ticker
                metrics["interval"] = interval
                metrics["strategy"] = strategy_name
                metrics["filename"] = filename

                all_results.append(metrics)

                # zapis per-run (opcjonalnie)
                out_csv = os.path.join(
                    RESULTS_DIR,
                    f"{filename.replace('.csv', '')}_{strategy_name}_backtest.csv",
                )
                results_df.to_csv(out_csv, sep=";")
            except Exception as e:
                print(f"❌ Błąd przy pliku {filename} [{strategy_name}]: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        summary = summary.sort_values(by="sharpe", ascending=False)

        print("\n📊 Wyniki zbiorcze (posortowane po Sharpe):")
        # wyświetl z ograniczoną precyzją
        with pd.option_context("display.float_format", "{:.6f}".format):
            print(summary.to_string(index=False))

        out_summary = os.path.join(RESULTS_DIR, "batch_results.csv")
        summary.to_csv(out_summary, index=False, sep=";")
        print(f"\n💾 Zapisano zbiorcze wyniki do: {out_summary}")
    else:
        print("\n⚠️ Nie znaleziono żadnych prawidłowych plików do przetworzenia.")


if __name__ == "__main__":
    main()
