from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PRED_FILE = Path("data/output/predictions_multi_source_7days.csv")

def main():
    df = pd.read_csv(PRED_FILE)

    mae_silam = mean_absolute_error(df["pm25_obs"], df["silam_pm25"])
    rmse_silam = np.sqrt(mean_squared_error(df["pm25_obs"], df["silam_pm25"]))

    mae_cmaq = mean_absolute_error(df["pm25_obs"], df["cmaq_pm25_approx"])
    rmse_cmaq = np.sqrt(mean_squared_error(df["pm25_obs"], df["cmaq_pm25_approx"]))

    mae_ai = mean_absolute_error(df["pm25_obs"], df["pm25_pred_corrected"])
    rmse_ai = np.sqrt(mean_squared_error(df["pm25_obs"], df["pm25_pred_corrected"]))

    print("=== ĐÁNH GIÁ MODEL HYBRID 7 NGÀY ===")
    print(f"SILAM raw  - MAE: {mae_silam:.3f} | RMSE: {rmse_silam:.3f}")
    print(f"CMAQ raw   - MAE: {mae_cmaq:.3f} | RMSE: {rmse_cmaq:.3f}")
    print(f"AI hybrid  - MAE: {mae_ai:.3f} | RMSE: {rmse_ai:.3f}")

if __name__ == "__main__":
    main()
