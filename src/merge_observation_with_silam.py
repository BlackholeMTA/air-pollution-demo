import pandas as pd
import numpy as np

OBS_PATH = "data/processed/observation_7days_all_stations_v2.csv"
SILAM_PATH = "data/processed/silam_7days_all_stations.csv"
OUT_PATH = "data/processed/obs_silam_7days.csv"

obs = pd.read_csv(OBS_PATH)
silam = pd.read_csv(SILAM_PATH)

obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
silam["timestamp"] = pd.to_datetime(silam["timestamp"], errors="coerce")

# Chỉ giữ các cột SILAM thật sự cần
silam_keep = [
    "timestamp", "station_id",
    "silam_pm25", "silam_pm10",
    "silam_o3", "silam_so2", "silam_co", "silam_no2",
    "silam_no", "silam_nox",
    "silam_lat_idx", "silam_lon_idx",
    "silam_grid_lat", "silam_grid_lon"
]
silam_keep = [c for c in silam_keep if c in silam.columns]
silam = silam[silam_keep].copy()

df = pd.merge(
    obs,
    silam,
    on=["timestamp", "station_id"],
    how="left"
)

df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("ĐÃ LƯU:", OUT_PATH)
print("Số dòng:", len(df))
print("Cột:", df.columns.tolist())

print("\nĐộ phủ SILAM sau khi ghép:")
for station_id, g in df.groupby("station_id"):
    print(f"\n== {station_id} ==")
    print("rows =", len(g))
    for c in ["silam_pm25","silam_pm10","silam_o3","silam_so2","silam_co","silam_no2"]:
        if c in g.columns:
            print(f"{c}: {g[c].notna().sum()}")
