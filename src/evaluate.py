import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test, scaler, target_idx, num_features, name="Model"):
    """
    Evaluates model predictions, ensuring physical bounds and ignoring near-zero rainfall in MAPE.
    Returns inverse-transformed true values, predicted values, and metrics dictionary.
    """
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform
    dummy_pred = np.zeros((len(y_pred_scaled), num_features))
    dummy_pred[:, target_idx] = y_pred_scaled
    y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]

    dummy_true = np.zeros((len(y_test), num_features))
    dummy_true[:, target_idx] = y_test
    y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

    # Clip negative predictions to 0 (rainfall can't be negative)
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)

    # Masked MAPE (ignore near-zero true values to avoid infinite MAPE)
    mask = y_true > 0.1
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 0.1))) * 100
    else:
        mape = 0.0
        
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name} — Test Results")
    print(f"  MAPE (masked): {mape:.2f}%")
    print(f"  MAE:           {mae:.4f} mm/hr")
    print(f"  RMSE:          {rmse:.4f} mm/hr")
    
    return y_true, y_pred, {'mape': mape, 'mae': mae, 'rmse': rmse}
