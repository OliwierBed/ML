# prepare_full_ensemble.py
import os
import pandas as pd

# Ścieżki – zmodyfikuj jeśli masz inne
PROCESSED_DIR = "data-pipelines/feature_stores/data/processed"
ENSEMBLE_DIR = os.path.join(PROCESSED_DIR, "ensemble")

def find_matching_input_file(ticker, interval):
    for fname in os.listdir(PROCESSED_DIR):
        if fname.lower().endswith(".csv") and ticker in fname and f"_{interval}_" in fname and "_ensemble" not in fname:
            return os.path.join(PROCESSED_DIR, fname)
    return None

def main():
    if not os.path.exists(ENSEMBLE_DIR):
        print(f"❌ Brak katalogu ENSEMBLE: {ENSEMBLE_DIR}")
        return

    files = [f for f in os.listdir(ENSEMBLE_DIR) if f.endswith("_ensemble.csv")]
    if not files:
        print("⚠️ Brak plików ensemble.")
        return

    print(f"📁 Używam katalogu ENSEMBLE: {ENSEMBLE_DIR}\n")
    for fname in files:
        print(f"🔁 Przetwarzam: {fname}")
        ens_path = os.path.join(ENSEMBLE_DIR, fname)
        ens_df = pd.read_csv(ens_path, sep=";")

        # Wyciągnij ticker i interval
        base = fname.replace("_ensemble.csv", "")
        try:
            ticker, interval = base.split("_", 1)
        except Exception:
            print(f"❌ Nie rozpoznano ticker/interwał z nazwy: {fname}")
            continue

        # Szukaj pasującego surowego pliku
        input_path = find_matching_input_file(ticker, interval)
        if not input_path:
            print(f"❌ Brak pliku z danymi OHLCV dla: {ticker} {interval}")
            continue

        # Wczytaj oryginalny plik z danymi i wskaźnikami
        orig_df = pd.read_csv(input_path, sep=";")
        orig_df.columns = [c.lower() for c in orig_df.columns]
        if "date" in orig_df.columns:
            orig_df["date"] = pd.to_datetime(orig_df["date"])
        if "datetime" in orig_df.columns:
            orig_df["datetime"] = pd.to_datetime(orig_df["datetime"])

        # Wczytaj ensemble – zamień na lowercase
        ens_df.columns = [c.lower() for c in ens_df.columns]
        if "date" in ens_df.columns:
            ens_df["date"] = pd.to_datetime(ens_df["date"])

        # MERGE: nadpisujemy/uzupełniamy kolumnę 'signal' wartościami z ensemble
        out_df = orig_df.drop(columns=["signal"], errors="ignore").merge(
            ens_df[["date", "signal"]],
            on="date",
            how="left"
        )

        # Jeśli w merge powstały dwie kolumny, bierzemy tę z ensemble
        if "signal_y" in out_df.columns:
            out_df["signal"] = out_df["signal_y"]
            out_df = out_df.drop(columns=["signal_x", "signal_y"])
        elif "signal" in out_df.columns:
            pass
        else:
            print(f"❌ Brak kolumny 'signal' po mergu dla {fname}")
            continue

        # Jeśli ensemble nie pokrywa wszystkich dat, to zostają NaN – zamień na 0 (lub jak chcesz)
        out_df["signal"] = out_df["signal"].fillna(0)

        # Walidacja: wymagane kolumny (np. close, open, high, low, volume)
        required_cols = {"close", "open", "high", "low", "volume"}
        if not required_cols.issubset(set(out_df.columns)):
            print(f"❌ Plik {fname} po mergu nie zawiera wszystkich wymaganych kolumn: {required_cols - set(out_df.columns)}")
            continue

        # Zapisz gotowy plik do katalogu ENSEMBLE
        out_path = os.path.join(ENSEMBLE_DIR, fname.replace("_ensemble.csv", "_ensemble_full.csv"))
        out_df.to_csv(out_path, sep=";", index=False)
        print(f"✅ Utworzono plik: {out_path}")

    print("\n🏁 Gotowe! Pliki *_ensemble_full.csv są gotowe do backtestu.")

if __name__ == "__main__":
    main()
