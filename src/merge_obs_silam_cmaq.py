import pandas as pd

OBS_SILAM_PATH = "data/processed/obs_silam_7days.csv"
CMAQ_PATH = "data/processed/cmaq_7days_all_stations.csv"
OUT_PATH = "data/processed/final_training_table_7days.csv"

obs_silam = pd.read_csv(OBS_SILAM_PATH)
cmaq = pd.read_csv(CMAQ_PATH)

obs_silam["timestamp"] = pd.to_datetime(obs_silam["timestamp"], errors="coerce")
cmaq["timestamp"] = pd.to_datetime(cmaq["timestamp"], errors="coerce")

# chỉ giữ các cột CMAQ cần thiết
cmaq_keep = [
    "timestamp", "station_id",
    "cmaq_row", "cmaq_col",
    "cmaq_o3", "cmaq_so2", "cmaq_co", "cmaq_no2", "cmaq_no", "cmaq_nox",
    "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"
]
cmaq_keep = [c for c in cmaq_keep if c in cmaq.columns]
cmaq = cmaq[cmaq_keep].copy()

df = pd.merge(
    obs_silam,
    cmaq,
    on=["timestamp", "station_id"],
    how="left"
)

df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("ĐÃ LƯU:", OUT_PATH)
print("Số dòng:", len(df))
print("Cột:", df.columns.tolist())

print("\nSố dòng theo trạm:")
print(df["station_id"].value_counts(dropna=False))

print("\nĐộ phủ một số cột chính:")
for c in [
    "pm25_obs", "pm10_obs", "o3_obs", "so2_obs", "co_obs", "no2_obs",
    "silam_pm25", "silam_pm10", "silam_o3", "silam_so2", "silam_co", "silam_no2",
    "cmaq_o3", "cmaq_so2", "cmaq_co", "cmaq_no2"
]:
    if c in df.columns:
        print(c, ":", df[c].notna().sum())
