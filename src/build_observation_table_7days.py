import os
import glob
import pandas as pd

INPUT_DIR = "data/processed/stations_merged"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMN_MAP = {
    "timestamp": "timestamp",
    "station_id": "station_id",
    "vn_aqi": "vn_aqi",
    "pm10_obs": "pm10_obs",
    "pm25_obs": "pm25_obs",
    "o3_obs": "o3_obs",
    "so2_obs": "so2_obs",
    "co_obs": "co_obs",
    "no2_obs": "no2_obs",
    "nox_obs": "nox_obs",
    "no_obs": "no_obs",

    "NO2(µg/Nm3)": "no2_obs",
    "NOx(µg/Nm3)": "nox_obs",
    "NO(µg/Nm3)": "no_obs",
    "SO2(µg/Nm3)": "so2_obs",
    "O3(µg/Nm3)": "o3_obs",
    "CO(mg/Nm3)": "co_obs",

    "Nhiệt độ(oC)": "temp",
    "Độ ẩm(%)": "rh",
    "Tốc độ gió(m/s)": "wind_speed",
    "Hướng gió(Degree)": "wind_dir",
    "Áp suất khí quyển(hPa)": "pressure",
    "Radiation(W/m2)": "radiation",

    "datetime_raw": "datetime_raw",
    "STT": "stt"
}

KEEP_COLUMNS = [
    "timestamp", "station_id", "datetime_raw", "stt",
    "vn_aqi",
    "pm25_obs", "pm10_obs", "o3_obs", "so2_obs", "co_obs", "no2_obs", "nox_obs", "no_obs",
    "temp", "rh", "wind_speed", "wind_dir", "pressure", "radiation"
]

def normalize_one_file(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})

    # nếu nox chưa có mà có no + no2 thì tự tính
    if "nox_obs" not in df.columns and "no_obs" in df.columns and "no2_obs" in df.columns:
        df["nox_obs"] = df["no_obs"].fillna(0) + df["no2_obs"].fillna(0)

    # chuẩn hóa timestamp
    if "timestamp" not in df.columns:
        raise ValueError(f"File {path} không có cột timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # chỉ giữ cột cần thiết nếu có
    use_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[use_cols].copy()

    # gộp trùng timestamp trong 1 trạm nếu có
    df = df.sort_values(["timestamp"])
    if "station_id" in df.columns:
        df = df.groupby(["station_id", "timestamp"], as_index=False).first()
    else:
        df = df.groupby(["timestamp"], as_index=False).first()

    return df

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*_merged_7days.csv"))
    if not files:
        raise ValueError("Không tìm thấy file merged CSV nào")

    frames = []
    for f in files:
        print(f"Đang đọc: {f}")
        df = normalize_one_file(f)
        frames.append(df)
        print(f"  -> rows={len(df)}, cols={df.columns.tolist()}")

    obs = pd.concat(frames, ignore_index=True)
    obs = obs.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    out_csv = os.path.join(OUTPUT_DIR, "observation_7days_all_stations.csv")
    obs.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\nĐã lưu:", out_csv)
    print("Số dòng:", len(obs))
    print("Cột:", obs.columns.tolist())

    print("\nĐộ phủ theo trạm:")
    for station_id, g in obs.groupby("station_id"):
        print(f"\n== {station_id} ==")
        print("rows =", len(g))
        for c in ["pm25_obs","pm10_obs","o3_obs","so2_obs","co_obs","no2_obs","nox_obs","no_obs","temp","rh","wind_speed","wind_dir","pressure","radiation"]:
            if c in g.columns:
                print(f"{c}: {g[c].notna().sum()}")

if __name__ == "__main__":
    main()
