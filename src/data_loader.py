import numpy as np
import pandas as pd
import os

def load_data(filepath):
    """
    Loads dataset from a parquet or csv file.
    """
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        return pd.read_csv(filepath)

def make_sequences(data, n_past, n_future, target_idx=0):
    """
    Creates overlapping windows for time series forecasting.
    data: 2D numpy array of shape (samples, features)
    n_past: number of past time steps to use as input
    n_future: how many steps ahead to forecast (usually 1 for nowcasting)
    target_idx: column index of the target variable
    """
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i + n_future - 1, target_idx])
    return np.array(X), np.array(y)
