import os
import joblib
import pandas as pd
import numpy as np

IN_PATH = "data/processed/final_training_table_with_features_7days.csv"
OUT_PATH = "data/output/predictions_6pollutants_7days.csv"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "pm25": {
        "model_path": "data/output/xgb_pm25_model.pkl",
        "features": [
            "silam_pm25", "silam_pm10",
            "pm25_obs_lag_1", "pm25_obs_lag_3", "pm25_obs_lag_6", "pm25_obs_lag_12", "pm25_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "pm10": {
        "model_path": "data/output/xgb_pm10_model.pkl",
        "features": [
            "silam_pm10", "silam_pm25",
            "pm10_obs_lag_1", "pm10_obs_lag_3", "pm10_obs_lag_6", "pm10_obs_lag_12", "pm10_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "o3": {
        "model_path": "data/output/xgb_o3_model.pkl",
        "features": [
            "silam_o3", "cmaq_o3",
            "o3_obs_lag_1", "o3_obs_lag_3", "o3_obs_lag_6", "o3_obs_lag_12", "o3_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "so2": {
        "model_path": "data/output/xgb_so2_model.pkl",
        "features": [
            "silam_so2", "cmaq_so2",
            "so2_obs_lag_1", "so2_obs_lag_3", "so2_obs_lag_6", "so2_obs_lag_12", "so2_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "co": {
        "model_path": "data/output/xgb_co_model.pkl",
        "features": [
            "silam_co", "cmaq_co",
            "co_obs_lag_1", "co_obs_lag_3", "co_obs_lag_6", "co_obs_lag_12", "co_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "no2": {
        "model_path": "data/output/xgb_no2_model.pkl",
        "features": [
            "silam_no2", "cmaq_no2", "cmaq_no", "cmaq_nox",
            "no2_obs_lag_1", "no2_obs_lag_3", "no2_obs_lag_6", "no2_obs_lag_12", "no2_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    }
}

def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for pollutant, cfg in MODEL_CONFIGS.items():
        model_path = cfg["model_path"]
        feature_cols = [c for c in cfg["features"] if c in df.columns]

        pred_col = f"{pollutant}_pred"
        df[pred_col] = np.nan

        if not os.path.exists(model_path):
            print(f"[SKIP] Không tìm thấy model: {model_path}")
            continue

        model = joblib.load(model_path)

        valid_mask = df[feature_cols].notna().all(axis=1)
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            print(f"[SKIP] {pollutant}: không có dòng nào đủ feature để predict")
            continue

        df.loc[valid_mask, pred_col] = model.predict(df.loc[valid_mask, feature_cols])
        print(f"[OK] {pollutant}: đã predict {n_valid} dòng")

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("\nĐÃ LƯU:", OUT_PATH)
    print("Số dòng:", len(df))

    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    print("\nĐộ phủ các cột dự báo:")
    for c in pred_cols:
        print(c, ":", df[c].notna().sum())

if __name__ == "__main__":
    main()
