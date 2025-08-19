import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    pred = pred.reshape(-1, pred.shape[-1])
    true = true.reshape(-1, true.shape[-1])
    pred_mean = pred.mean(axis=0)
    true_mean = true.mean(axis=0)

    num = np.sum((pred - pred_mean) * (true - true_mean), axis=0)
    den = np.sqrt(np.sum((pred - pred_mean) ** 2, axis=0) * np.sum((true - true_mean) ** 2, axis=0))
    corr = num / den

    return np.mean(corr)  # hasil akhir adalah scalar antara -1 dan 1


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, corr
