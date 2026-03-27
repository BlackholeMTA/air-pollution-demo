from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PRED_FILE = Path("data/output/predictions_cmaq_hanoi.csv")

def main():
    df = pd.read_csv(PRED_FILE)

    mae_raw = mean_absolute_error(df["pm25_obs"], df["pm25_cmaq_approx"])
    rmse_raw = np.sqrt(mean_squared_error(df["pm25_obs"], df["pm25_cmaq_approx"]))

    mae_ai = mean_absolute_error(df["pm25_obs"], df["pm25_pred_corrected"])
    rmse_ai = np.sqrt(mean_squared_error(df["pm25_obs"], df["pm25_pred_corrected"]))

    print("=== ĐÁNH GIÁ MÔ HÌNH ===")
    print(f"MAE raw      : {mae_raw:.3f}")
    print(f"RMSE raw     : {rmse_raw:.3f}")
    print(f"MAE corrected: {mae_ai:.3f}")
    print(f"RMSE corrected: {rmse_ai:.3f}")

    print("\nCải thiện:")
    print(f"MAE giảm: {mae_raw - mae_ai:.3f}")
    print(f"RMSE giảm: {rmse_raw - rmse_ai:.3f}")

if __name__ == "__main__":
    main()
