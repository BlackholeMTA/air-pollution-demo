import os
import re
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset

BASE_DIR = "data/raw/silam"
META_PATH = "data/raw/stations/station_metadata.csv"
OUT_PATH = "data/processed/silam_7days_all_stations.csv"

DATES = ["20251225", "20251226", "20251227", "20251228", "20251229", "20251230", "20251231"]

# Các biến khí cần lấy từ file thường YYYYMMDDHH.nc4
GAS_VARS = {
    "silam_o3": "cnc_O3_gas",
    "silam_so2": "cnc_SO2_gas",
    "silam_co": "cnc_CO_gas",
    "silam_no2": "cnc_NO2_gas",
    "silam_no": "cnc_NO_gas",
}

# Các biến PM cần lấy từ file PM_YYYYMMDDHH.nc4
PM_VARS = {
    "silam_pm10": "cnc_PM10",
    "silam_pm25": "cnc_PM2_5",
}


def read_station_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_PATH)
    required = {"station_id", "station_name", "lat", "lon"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"station_metadata.csv thiếu cột: {missing}")
    return meta[["station_id", "station_name", "lat", "lon"]].copy()


def find_nearest_grid_1d(lat_arr, lon_arr, target_lat, target_lon):
    """
    SILAM hiện có lat(195,), lon(190,) dạng 1D.
    Tìm ô lưới gần nhất theo từng chiều.
    """
    lat_idx = int(np.abs(lat_arr - target_lat).argmin())
    lon_idx = int(np.abs(lon_arr - target_lon).argmin())
    return lat_idx, lon_idx


def safe_get_scalar(var, time_idx, lat_idx, lon_idx):
    try:
        return float(var[time_idx, lat_idx, lon_idx])
    except Exception:
        return np.nan


def list_gas_files_for_day(day: str):
    """
    Chỉ lấy file khí đúng ngày, bỏ toàn bộ Zone.Identifier, ctl, AQI...
    Chấp nhận các file như:
      2025122500.nc4, 2025122501.nc4, ...
    """
    day_dir = os.path.join(BASE_DIR, day)
    if not os.path.exists(day_dir):
        return []

    all_files = sorted(glob.glob(os.path.join(day_dir, "*.nc4")))
    gas_files = []

    pattern = re.compile(rf"^{day}(\d{{2}})\.nc4$")
    for f in all_files:
        fname = os.path.basename(f)

        # bỏ file PM và AQI
        if fname.startswith("PM_") or fname.startswith("AQI"):
            continue

        # chỉ lấy file đúng pattern YYYYMMDDHH.nc4
        m = pattern.match(fname)
        if m:
            gas_files.append((f, m.group(1)))  # (path, hour)

    return gas_files


def list_pm_files_for_day(day: str):
    """
    Chỉ lấy file PM đúng ngày:
      PM_YYYYMMDDHH.nc4
    """
    day_dir = os.path.join(BASE_DIR, day)
    if not os.path.exists(day_dir):
        return []

    all_files = sorted(glob.glob(os.path.join(day_dir, "PM_*.nc4")))
    pm_files = []

    pattern = re.compile(rf"^PM_{day}(\d{{2}})\.nc4$")
    for f in all_files:
        fname = os.path.basename(f)
        m = pattern.match(fname)
        if m:
            pm_files.append((f, m.group(1)))  # (path, hour)

    return pm_files


def extract_from_one_file(nc_path, stations, var_map, timestamp):
    rows = []

    ds = Dataset(nc_path)

    if "lat" not in ds.variables or "lon" not in ds.variables:
        ds.close()
        raise ValueError(f"File {nc_path} không có biến lat/lon")

    lat_arr = ds.variables["lat"][:]
    lon_arr = ds.variables["lon"][:]

    if "time" not in ds.variables:
        time_len = 1
    else:
        time_len = len(ds.variables["time"][:])

    # Mặc định dùng time_idx = 0 vì file của bạn đang là từng giờ/file
    time_idx = 0
    if time_len < 1:
        ds.close()
        return pd.DataFrame()

    for _, st in stations.iterrows():
        lat_idx, lon_idx = find_nearest_grid_1d(lat_arr, lon_arr, st["lat"], st["lon"])

        row = {
            "timestamp": timestamp,
            "station_id": st["station_id"],
            "station_name": st["station_name"],
            "station_lat": st["lat"],
            "station_lon": st["lon"],
            "silam_lat_idx": lat_idx,
            "silam_lon_idx": lon_idx,
            "silam_grid_lat": float(lat_arr[lat_idx]),
            "silam_grid_lon": float(lon_arr[lon_idx]),
            "source_file": os.path.basename(nc_path),
        }

        for out_col, var_name in var_map.items():
            if var_name in ds.variables:
                row[out_col] = safe_get_scalar(ds.variables[var_name], time_idx, lat_idx, lon_idx)
            else:
                row[out_col] = np.nan

        rows.append(row)

    ds.close()
    return pd.DataFrame(rows)


def extract_gas_for_all_days(stations):
    frames = []

    for day in DATES:
        gas_files = list_gas_files_for_day(day)
        print(f"\n[DAY {day}] Gas files hợp lệ: {len(gas_files)}")

        for gas_path, hh in gas_files:
            timestamp = pd.to_datetime(f"{day} {hh}:00:00", format="%Y%m%d %H:%M:%S")
            try:
                df = extract_from_one_file(gas_path, stations, GAS_VARS, timestamp)
                frames.append(df)
                print(f"  [OK] {os.path.basename(gas_path)} -> rows={len(df)}")
            except Exception as e:
                print(f"  [LỖI] {os.path.basename(gas_path)}: {e}")

    if frames:
        gas_all = pd.concat(frames, ignore_index=True)
        gas_all = gas_all.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        return gas_all

    return pd.DataFrame()


def extract_pm_for_all_days(stations):
    frames = []

    for day in DATES:
        pm_files = list_pm_files_for_day(day)
        print(f"\n[DAY {day}] PM files hợp lệ: {len(pm_files)}")

        for pm_path, hh in pm_files:
            timestamp = pd.to_datetime(f"{day} {hh}:00:00", format="%Y%m%d %H:%M:%S")
            try:
                df = extract_from_one_file(pm_path, stations, PM_VARS, timestamp)
                frames.append(df)
                print(f"  [OK] {os.path.basename(pm_path)} -> rows={len(df)}")
            except Exception as e:
                print(f"  [LỖI] {os.path.basename(pm_path)}: {e}")

    if frames:
        pm_all = pd.concat(frames, ignore_index=True)
        pm_all = pm_all.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        return pm_all

    return pd.DataFrame()


def merge_pm_gas(pm_all: pd.DataFrame, gas_all: pd.DataFrame) -> pd.DataFrame:
    if pm_all.empty and gas_all.empty:
        return pd.DataFrame()

    key_cols = ["timestamp", "station_id"]

    keep_pm_cols = [
        "timestamp", "station_id", "station_name",
        "station_lat", "station_lon",
        "silam_lat_idx", "silam_lon_idx",
        "silam_grid_lat", "silam_grid_lon",
        "source_file",
        "silam_pm10", "silam_pm25"
    ]
    keep_pm_cols = [c for c in keep_pm_cols if c in pm_all.columns]
    pm_all = pm_all[keep_pm_cols].copy() if not pm_all.empty else pd.DataFrame(columns=key_cols)

    keep_gas_cols = [
        "timestamp", "station_id",
        "silam_o3", "silam_so2", "silam_co", "silam_no2", "silam_no"
    ]
    keep_gas_cols = [c for c in keep_gas_cols if c in gas_all.columns]
    gas_all = gas_all[keep_gas_cols].copy() if not gas_all.empty else pd.DataFrame(columns=key_cols)

    if not pm_all.empty and not gas_all.empty:
        silam = pd.merge(pm_all, gas_all, on=key_cols, how="outer")
    elif not pm_all.empty:
        silam = pm_all.copy()
    else:
        silam = gas_all.copy()

    if "silam_no" in silam.columns and "silam_no2" in silam.columns:
        silam["silam_nox"] = silam["silam_no"].fillna(0) + silam["silam_no2"].fillna(0)

    silam = silam.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return silam


def main():
    stations = read_station_metadata()

    print("=== STATIONS ===")
    print(stations)

    gas_all = extract_gas_for_all_days(stations)
    pm_all = extract_pm_for_all_days(stations)

    print("\n=== TỔNG HỢP SAU KHI TRÍCH ===")
    print("Gas rows:", len(gas_all))
    print("PM rows :", len(pm_all))

    silam = merge_pm_gas(pm_all, gas_all)
    silam.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("ĐÃ LƯU:", OUT_PATH)
    print("==============================")
    print("Số dòng:", len(silam))
    print("Cột:", silam.columns.tolist())

    if not silam.empty:
        print("\nĐộ phủ theo trạm:")
        for station_id, g in silam.groupby("station_id"):
            print(f"\n== {station_id} ==")
            print("rows =", len(g))
            for c in ["silam_pm25", "silam_pm10", "silam_o3", "silam_so2", "silam_co", "silam_no2", "silam_no", "silam_nox"]:
                if c in g.columns:
                    print(f"{c}: {g[c].notna().sum()}")
    else:
        print("Không trích được dữ liệu SILAM nào.")


if __name__ == "__main__":
    main()
