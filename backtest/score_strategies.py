import os
import pandas as pd

try:
    from config.load import load_config
except Exception:
    from omegaconf import OmegaConf
    def load_config(path: str = "config/config.yaml"):
        return OmegaConf.load(path)


def main():
    cfg = load_config()

    # POBIERANIE ŚCIEŻEK Z CONFIGA!
    RESULTS_DIR = getattr(cfg.paths, "results", "backtest/results")
    in_path = os.path.join(RESULTS_DIR, "batch_results.csv")
    out_top_overall = os.path.join(RESULTS_DIR, "top_overall.csv")
    out_top_bucket = os.path.join(RESULTS_DIR, "top_per_bucket.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {in_path}. Najpierw uruchom: python -m backtest.runner_batch"
        )

    metric = getattr(cfg.scoring, "metric", "sharpe")
    top_n = int(getattr(cfg.scoring, "top_n", 20))
    direction = getattr(cfg.scoring, "direction", "desc").lower()
    ascending = direction != "desc"

    # Twoje CSV jest rozdzielane średnikiem!
    df = pd.read_csv(in_path, sep=";")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if metric not in df.columns:
        raise ValueError(
            f"Wybrana metryka '{metric}' nie istnieje w batch_results.csv. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    top_overall = (
        df.sort_values(by=metric, ascending=ascending)
          .head(top_n)
          .reset_index(drop=True)
    )

    top_bucket = (
        df.sort_values(by=metric, ascending=ascending)
          .groupby(["ticker", "interval"], as_index=False)
          .head(1)
          .reset_index(drop=True)
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    top_overall.to_csv(out_top_overall, index=False)
    top_bucket.to_csv(out_top_bucket, index=False)

    print(f"✅ Zapisano TOP {top_n} strategii (globalnie): {out_top_overall}")
    print(f"✅ Zapisano TOP 1 per (ticker, interval):     {out_top_bucket}")
    print("\n📌 Użyta metryka:", metric)
    print("📌 Kierunek sortowania:", "malejąco" if not ascending else "rosnąco")


if __name__ == "__main__":
    main()
