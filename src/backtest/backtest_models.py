# src/backtest/backtest_models.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
# directional accuracy = percent of times sign(predicted_change)==sign(actual_change)
def directional_accuracy(y_true, y_pred):
    actual_change = np.sign(np.diff(y_true, prepend=y_true[0]))
    pred_change   = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    return np.mean(actual_change == pred_change)

def make_sequences(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :-1])
        y.append(data[i + window_size - 1, 3])  # assuming 'c' (close) is col index 3
    return np.array(X), np.array(y)

def backtest_model(
    model_path: str,
    scaler,
    test_csv: str,
    window_size: int = 10,
    results_dir: str = "data/backtest"
):
    # 1. Load test data
    df_test = pd.read_csv(
        test_csv,
        parse_dates=['datetime'],
        index_col='datetime',
        infer_datetime_format=True
    )
    values = df_test.values

    # 2. Scale features
    data_scaled = scaler.transform(values)  # scaler fitted on train :contentReference[oaicite:1]{index=1}

    # 3. Create sequences
    X_test, y_test = make_sequences(data_scaled, window_size)

    # 4. Load model & predict
    model = load_model(model_path)
    y_pred_scaled = model.predict(X_test, verbose=0)  # uses model.predict() :contentReference[oaicite:2]{index=2}

    # 5. Invert scaling for predictions & ground truth
    # Reconstruct full feature array for inverse_transform
    dummy = np.zeros((len(y_pred_scaled), values.shape[1]))
    dummy[:, 3] = y_pred_scaled.flatten()  
    inv_pred = scaler.inverse_transform(dummy)[:, 3]
    dummy[:, 3] = y_test
    inv_true = scaler.inverse_transform(dummy)[:, 3]

    # 6. Compute metrics
    mse  = mean_squared_error(inv_true, inv_pred)      # :contentReference[oaicite:3]{index=3}
    mae  = mean_absolute_error(inv_true, inv_pred)     # analogous :contentReference[oaicite:4]{index=4}
    da   = directional_accuracy(inv_true, inv_pred)

    # 7. Save results
    os.makedirs(results_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(model_path))[0]
    df_out = pd.DataFrame({
        'true': inv_true,
        'pred': inv_pred
    }, index=df_test.index[window_size:])
    df_out.to_csv(f"{results_dir}/{base}_predictions.csv")
    with open(f"{results_dir}/{base}_metrics.txt", 'w') as f:
        f.write(f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nDirAcc: {da:.4f}\n")

    print(f"Backtested {base}: MSE={mse:.4f}, MAE={mae:.4f}, DirAcc={da:.4f}")
    return mse, mae, da
