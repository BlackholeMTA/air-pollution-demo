from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset

SILAM_DIR = Path("data/raw/silam/20251227")
STATION_FILE = Path("data/raw/stations/station_metadata.csv")
OUT_FILE = Path("data/processed/silam_station_series.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def find_nearest_idx(arr, value):
    return int(np.abs(arr - value).argmin())

def parse_time_from_filename(filename: str):
    # ví dụ: PM_2025122700.nc4 -> 2025-12-27 00:00:00
    stem = Path(filename).stem  # PM_2025122700
    time_part = stem.replace("PM_", "")
    return pd.to_datetime(time_part, format="%Y%m%d%H")

def main():
    stations = pd.read_csv(STATION_FILE)

    files = sorted([
        f for f in SILAM_DIR.glob("PM_*.nc4")
        if ":Zone.Identifier" not in f.name
    ])

    print(f"So file SILAM PM tim thay: {len(files)}")
    if not files:
        print("Khong tim thay file PM_*.nc4")
        return

    rows = []

    for file_path in files:
        ts = parse_time_from_filename(file_path.name)
        ds = Dataset(file_path)

        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        pm25 = ds.variables["cnc_PM2_5"][:]   # (1, lat, lon)
        pm10 = ds.variables["cnc_PM10"][:]    # (1, lat, lon)
        aqi = ds.variables["AQI"][:]          # (1, lat, lon)

        for _, row in stations.iterrows():
            lat_idx = find_nearest_idx(lats, row["lat"])
            lon_idx = find_nearest_idx(lons, row["lon"])

            rows.append({
                "timestamp": ts,
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "silam_pm25": float(pm25[0, lat_idx, lon_idx]),
                "silam_pm10": float(pm10[0, lat_idx, lon_idx]),
                "silam_aqi": float(aqi[0, lat_idx, lon_idx]),
            })

        ds.close()

    out_df = pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    print(out_df.head(15))
    print(f"\nShape: {out_df.shape}")

    out_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
