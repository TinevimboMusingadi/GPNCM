import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def fit_and_scale_splits(df_train, df_val, df_test, feature_cols, scaler_save_path=None):
    """
    Fits a StandardScaler on the training set and scales the train, val, and test data.
    Saves the scaler to disk to prevent data leakage and for later inference.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    train_scaled = scaler.transform(df_train[feature_cols])
    val_scaled   = scaler.transform(df_val[feature_cols])
    test_scaled  = scaler.transform(df_test[feature_cols])

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return train_scaled, val_scaled, test_scaled, scaler
