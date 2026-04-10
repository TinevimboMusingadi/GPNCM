import numpy as np
import pandas as pd
import os
import yaml
from src.preprocessing import fit_and_scale_splits

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

def load_station_data(station_dir, n_past):
    """
    Orchestrates loading, scaling, and sequence creation for a specific station.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx
    """
    # 1. Load Global Config for features
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    feature_cols = cfg['data']['feature_cols']
    target_col = cfg['data']['target_col']
    n_future = cfg['data'].get('n_future', 1)
    target_idx = feature_cols.index(target_col)

    # 2. Load the pre-split files
    df_train = pd.read_parquet(os.path.join(station_dir, 'train.parquet'))
    df_val   = pd.read_parquet(os.path.join(station_dir, 'val.parquet'))
    df_test  = pd.read_parquet(os.path.join(station_dir, 'test.parquet'))

    # 3. Fit scaler on train and transform all
    # We don't save the scaler here as it's meant for the iterative training step
    train_sc, val_sc, test_sc, scaler = fit_and_scale_splits(df_train, df_val, df_test, feature_cols)

    # 4. Create Sequences
    X_train, y_train = make_sequences(train_sc, n_past, n_future, target_idx)
    X_val, y_val     = make_sequences(val_sc, n_past, n_future, target_idx)
    X_test, y_test   = make_sequences(test_sc, n_past, n_future, target_idx)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx
