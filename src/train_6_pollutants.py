import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

IN_PATH = "data/processed/final_training_table_with_features_7days.csv"
OUTPUT_DIR = "data/output"
OUT_METRICS = os.path.join(OUTPUT_DIR, "pollutant_metrics_6models.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_END = pd.Timestamp("2025-12-29 23:59:59")
TEST_START = pd.Timestamp("2025-12-30 00:00:00")

POLLUTANT_CONFIGS = {
    "pm25": {
        "target": "pm25_obs",
        "features": [
            "silam_pm25", "silam_pm10",
            "pm25_obs_lag_1", "pm25_obs_lag_3", "pm25_obs_lag_6", "pm25_obs_lag_12", "pm25_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "pm10": {
        "target": "pm10_obs",
        "features": [
            "silam_pm10", "silam_pm25",
            "pm10_obs_lag_1", "pm10_obs_lag_3", "pm10_obs_lag_6", "pm10_obs_lag_12", "pm10_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "o3": {
        "target": "o3_obs",
        "features": [
            "silam_o3", "cmaq_o3",
            "o3_obs_lag_1", "o3_obs_lag_3", "o3_obs_lag_6", "o3_obs_lag_12", "o3_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "so2": {
        "target": "so2_obs",
        "features": [
            "silam_so2", "cmaq_so2",
            "so2_obs_lag_1", "so2_obs_lag_3", "so2_obs_lag_6", "so2_obs_lag_12", "so2_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "co": {
        "target": "co_obs",
        "features": [
            "silam_co", "cmaq_co",
            "co_obs_lag_1", "co_obs_lag_3", "co_obs_lag_6", "co_obs_lag_12", "co_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    },
    "no2": {
        "target": "no2_obs",
        "features": [
            "silam_no2", "cmaq_no2", "cmaq_no", "cmaq_nox",
            "no2_obs_lag_1", "no2_obs_lag_3", "no2_obs_lag_6", "no2_obs_lag_12", "no2_obs_lag_24",
            "hour", "dayofweek", "month", "is_weekend",
            "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
        ]
    }
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nrmse(y_true, y_pred):
    r = rmse(y_true, y_pred)
    y_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    if y_range == 0:
        return np.nan
    return r / y_range


def train_one_pollutant(df, pollutant_name, cfg):
    target_col = cfg["target"]
    feature_cols = [c for c in cfg["features"] if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Thiếu cột target: {target_col}")

    need_cols = ["timestamp", "station_id", target_col] + feature_cols
    sub = df[need_cols].copy()

    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.dropna(subset=["timestamp", target_col])

    # chia train / test theo thời gian
    train_df = sub[sub["timestamp"] <= TRAIN_END].copy()
    test_df = sub[sub["timestamp"] >= TEST_START].copy()

    # chỉ giữ các hàng đủ feature
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    if len(train_df) < 20:
        raise ValueError(f"{pollutant_name}: train quá ít dòng ({len(train_df)})")
    if len(test_df) < 5:
        raise ValueError(f"{pollutant_name}: test quá ít dòng ({len(test_df)})")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = XGBRegressor(
        booster="gblinear",
        n_estimators=300,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_path = os.path.join(OUTPUT_DIR, f"xgb_{pollutant_name}_model.pkl")
    joblib.dump(model, model_path)

    result = {
        "pollutant": pollutant_name,
        "target": target_col,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "features_used": ", ".join(feature_cols),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "nrmse": nrmse(y_test, y_pred),
        "r2": float(r2_score(y_test, y_pred)),
        "model_path": model_path
    }

    return result


def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    results = []

    for pollutant_name, cfg in POLLUTANT_CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"ĐANG TRAIN: {pollutant_name}")
        print("=" * 70)

        try:
            res = train_one_pollutant(df, pollutant_name, cfg)
            results.append(res)
            print("[OK]")
            print(res)
        except Exception as e:
            print("[SKIP]", pollutant_name, "->", e)

    if results:
        out = pd.DataFrame(results)
        out.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
        print("\nĐÃ LƯU:", OUT_METRICS)
        print(out)
    else:
        print("Không train được model nào.")


if __name__ == "__main__":
    main()
