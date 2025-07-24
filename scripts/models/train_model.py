import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(processed_dir="data/processed", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    features = ["sma_20", "rsi_14", "atr_14", "obv", "bb_low", "bb_mid", "bb_high", 
                "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist"]
    target = "close"

    for file in os.listdir(processed_dir):
        if file.endswith("_indicators.csv"):
            ticker, interval = file.split("_")[0], file.split("_")[1]
            logging.info(f"Trenowanie modelu dla {ticker} ({interval})")
            df = pd.read_csv(os.path.join(processed_dir, file), sep=";")

            # Przygotowanie danych
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Trening modelu
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)

            # Predykcja i metryki
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Metryki dla {ticker} ({interval}): MSE={mse:.2f}, R2={r2:.2f}")

            # Zapis modelu
            model_path = os.path.join(model_dir, f"{ticker}_{interval}_model.pkl")
            joblib.dump(model, model_path)
            logging.info(f"Model zapisany do {model_path}")

if __name__ == "__main__":
    train_model()