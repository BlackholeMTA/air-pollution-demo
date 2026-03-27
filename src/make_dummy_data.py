import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

n = 200
time_index = pd.date_range("2026-03-20 00:00:00", periods=n, freq="H")

pm25_raw = 40 + 15 * np.sin(np.arange(n) / 10) + np.random.normal(0, 4, n)
pm25_obs = pm25_raw * 0.85 + 8 + np.random.normal(0, 3, n)

df = pd.DataFrame({
    "timestamp": time_index,
    "station_id": ["HN01"] * n,
    "pm25_raw": pm25_raw,
    "pm25_obs": pm25_obs
})

df["pm25_lag1"] = df["pm25_obs"].shift(1)
df["pm25_lag3"] = df["pm25_obs"].shift(3)
df["pm25_roll6"] = df["pm25_obs"].rolling(window=6, min_periods=1).mean()
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["target_pm25"] = df["pm25_obs"]

df = df.dropna().reset_index(drop=True)
df.to_csv(out_dir / "merged_features.csv", index=False)

print("Dummy data created at data/processed/merged_features.csv")
print(df.head())
