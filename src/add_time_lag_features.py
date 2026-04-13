import pandas as pd

IN_PATH = "data/processed/final_training_table_7days.csv"
OUT_PATH = "data/processed/final_training_table_with_features_7days.csv"

POLLUTANTS = ["pm25", "pm10", "o3", "so2", "co", "no2"]
LAGS = [1, 3, 6, 12, 24]

df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Sắp xếp chuẩn theo trạm và thời gian
df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

# Feature thời gian
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# Lag theo từng trạm, từng chất
for p in POLLUTANTS:
    col = f"{p}_obs"
    if col in df.columns:
        for lag in LAGS:
            df[f"{p}_obs_lag_{lag}"] = df.groupby("station_id")[col].shift(lag)

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("ĐÃ LƯU:", OUT_PATH)
print("Số dòng:", len(df))
print("Cột:", df.columns.tolist())

print("\nKiểm tra một số cột lag:")
check_cols = [c for c in df.columns if "lag" in c][:20]
print(check_cols)

print("\nĐộ phủ sau khi tạo lag:")
for c in [
    "pm25_obs_lag_1", "pm25_obs_lag_24",
    "pm10_obs_lag_1", "pm10_obs_lag_24",
    "o3_obs_lag_1", "o3_obs_lag_24",
    "so2_obs_lag_1", "so2_obs_lag_24",
    "co_obs_lag_1", "co_obs_lag_24",
    "no2_obs_lag_1", "no2_obs_lag_24",
]:
    if c in df.columns:
        print(c, ":", df[c].notna().sum())
