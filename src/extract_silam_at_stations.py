from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset

SILAM_FILE = Path("data/raw/silam/20251227/PM_2025122700.nc4")
STATION_FILE = Path("data/raw/stations/station_metadata.csv")
OUT_FILE = Path("data/processed/silam_station_pm25_2025122700.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def find_nearest_idx(arr, value):
    return int(np.abs(arr - value).argmin())

def main():
    ds = Dataset(SILAM_FILE)

    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    pm25 = ds.variables["cnc_PM2_5"][:]   # shape (1, lat, lon)
    pm10 = ds.variables["cnc_PM10"][:]    # shape (1, lat, lon)
    aqi = ds.variables["AQI"][:]          # shape (1, lat, lon)

    stations = pd.read_csv(STATION_FILE)

    rows = []
    for _, row in stations.iterrows():
        lat_idx = find_nearest_idx(lats, row["lat"])
        lon_idx = find_nearest_idx(lons, row["lon"])

        rows.append({
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

    out_df = pd.DataFrame(rows)
    print(out_df)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

    ds.close()

if __name__ == "__main__":
    main()
