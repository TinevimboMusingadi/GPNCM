import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def split_and_scale_data(df, feature_cols, target_col, train_frac=0.70, val_frac=0.15, scaler_save_path=None):
    """
    Chronologically splits data into train/val/test and scales features based on train set to prevent data leakage.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    train_scaled = scaler.transform(df_train[feature_cols])
    val_scaled   = scaler.transform(df_val[feature_cols])
    test_scaled  = scaler.transform(df_test[feature_cols])

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return train_scaled, val_scaled, test_scaled, scaler
