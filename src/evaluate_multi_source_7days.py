from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PRED_FILE = Path("data/output/predictions_multi_source_7days.csv")

def calc_nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    value_range = y_true.max() - y_true.min()
    if value_range == 0:
        return np.nan
    return rmse / value_range

def main():
    df = pd.read_csv(PRED_FILE)

    y_true = df["pm25_obs"]

    mae_silam = mean_absolute_error(y_true, df["silam_pm25"])
    rmse_silam = np.sqrt(mean_squared_error(y_true, df["silam_pm25"]))
    nrmse_silam = calc_nrmse(y_true, df["silam_pm25"])

    mae_cmaq = mean_absolute_error(y_true, df["cmaq_pm25_approx"])
    rmse_cmaq = np.sqrt(mean_squared_error(y_true, df["cmaq_pm25_approx"]))
    nrmse_cmaq = calc_nrmse(y_true, df["cmaq_pm25_approx"])

    mae_ai = mean_absolute_error(y_true, df["pm25_pred_corrected"])
    rmse_ai = np.sqrt(mean_squared_error(y_true, df["pm25_pred_corrected"]))
    nrmse_ai = calc_nrmse(y_true, df["pm25_pred_corrected"])

    print("=== ĐÁNH GIÁ MODEL HYBRID 7 NGÀY ===")
    print(f"SILAM raw  - MAE: {mae_silam:.3f} | RMSE: {rmse_silam:.3f} | NRMSE: {nrmse_silam:.3f}")
    print(f"CMAQ raw   - MAE: {mae_cmaq:.3f} | RMSE: {rmse_cmaq:.3f} | NRMSE: {nrmse_cmaq:.3f}")
    print(f"AI hybrid  - MAE: {mae_ai:.3f} | RMSE: {rmse_ai:.3f} | NRMSE: {nrmse_ai:.3f}")

if __name__ == "__main__":
    main()
