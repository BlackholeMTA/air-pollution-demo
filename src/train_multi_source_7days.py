from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = Path("data/processed/train_multi_source_hanoi_7days.csv")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUT_DIR / "xgb_multi_source_7days.pkl"
PRED_FILE = OUT_DIR / "predictions_multi_source_7days.csv"
PRED_FULL_FILE = OUT_DIR / "predictions_multi_source_7days_full.csv"

def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day

    station_map = {sid: i for i, sid in enumerate(sorted(df["station_id"].unique()))}
    df["station_code"] = df["station_id"].map(station_map)

    df = df.sort_values(["station_id", "timestamp"]).copy()

    df["pm25_lag1"] = df.groupby("station_id")["pm25_obs"].shift(1)
    df["pm25_lag3"] = df.groupby("station_id")["pm25_obs"].shift(3)
    df["pm25_lag6"] = df.groupby("station_id")["pm25_obs"].shift(6)
    df["pm25_roll6"] = (
        df.groupby("station_id")["pm25_obs"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["pm25_roll12"] = (
        df.groupby("station_id")["pm25_obs"]
        .rolling(12, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["pm10_lag1"] = df.groupby("station_id")["pm10_obs"].shift(1)
    df["pm10_roll6"] = (
        df.groupby("station_id")["pm10_obs"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "silam_pm25",
        "silam_pm10",
        "silam_aqi",
        "cmaq_pm25_approx",
        "pm25_lag1",
        "pm25_lag3",
        "pm25_lag6",
        "pm25_roll6",
        "pm25_roll12",
        "pm10_obs",
        "pm10_lag1",
        "pm10_roll6",
        "pm10_aqi",
        "pm25_aqi",
        "vn_aqi",
        "hour",
        "dayofweek",
        "day",
        "station_code",
    ]
    target_col = "pm25_obs"

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = xgb.XGBRegressor(
        booster="gbtree",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    test_df["pm25_pred_corrected"] = y_pred_test
    test_df.to_csv(PRED_FILE, index=False)

    df["pm25_pred_corrected"] = model.predict(df[feature_cols])
    df.to_csv(PRED_FULL_FILE, index=False)

    joblib.dump(model, MODEL_FILE)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved test predictions to {PRED_FILE}")
    print(f"Saved full predictions to {PRED_FULL_FILE}")

if __name__ == "__main__":
    main()
