import pandas as pd
from ml.inference.predict_lstm import load_model_bundle, forecast_next

class LSTMStrategy:
    def __init__(self, df, ticker, interval, models_dir="models", threshold=0.0):
        self.df = df.copy()
        self.ticker = ticker
        self.interval = interval
        self.models_dir = models_dir
        self.threshold = threshold
        self.model, self.scaler, self.meta = load_model_bundle(ticker, interval, models_dir)

    def generate_signals(self):
        # tu możesz przesuwać okno po df i prognozować *każdą* świecę (wolniej),
        # albo po prostu zrobić prognozę „następnej” i wstawić na koniec.
        closes = self.df["close"]
        preds = []
        for i in range(len(closes)):
            if i < self.meta["seq_len"]:
                preds.append(None)
                continue
            window = closes.iloc[:i]
            try:
                pred = forecast_next(window, self.model, self.scaler, self.meta)
            except Exception:
                pred = None
            preds.append(pred)
        self.df["pred"] = preds
        self.df["signal"] = 0
        mask = self.df["pred"].notna()
        self.df.loc[mask & (self.df["pred"] > self.df["close"]*(1+self.threshold)), "signal"] = 1
        self.df.loc[mask & (self.df["pred"] < self.df["close"]*(1-self.threshold)), "signal"] = -1
        return self.df[["date", "signal"]]
