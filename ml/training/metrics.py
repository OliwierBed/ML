import numpy as np

def rmse(y_true, y_pred): return np.sqrt(((y_true - y_pred) ** 2).mean())
def mae(y_true, y_pred): return np.abs(y_true - y_pred).mean()
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / y_true + 1e-9)) * 100

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
