from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset

SILAM_ROOT = Path("data/raw/silam")
STATION_FILE = Path("data/raw/stations/station_metadata.csv")
OUT_FILE = Path("data/processed/silam_station_series_7days.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Chuẩn 7 ngày
START_TS = pd.Timestamp("2025-12-25 00:00:00")
END_TS = pd.Timestamp("2025-12-31 23:00:00")

def find_nearest_idx(arr, value):
    return int(np.abs(arr - value).argmin())

def parse_valid_time_from_filename(filename: str):
    # PM_2025122700.nc4 -> 2025-12-27 00:00:00
    stem = Path(filename).stem
    time_part = stem.replace("PM_", "")
    return pd.to_datetime(time_part, format="%Y%m%d%H")

def parse_run_time_from_parent(folder_name: str):
    # thư mục 20251225 -> 2025-12-25 00:00:00
    return pd.to_datetime(folder_name, format="%Y%m%d")

def main():
    stations = pd.read_csv(STATION_FILE)

    raw_files = [
        f for f in SILAM_ROOT.rglob("PM_*.nc4")
        if ":Zone.Identifier" not in f.name
    ]

    print(f"So file SILAM PM tim thay truoc khi loc: {len(raw_files)}")

    # Bước 1: gom metadata file
    meta_rows = []
    for f in raw_files:
        try:
            valid_ts = parse_valid_time_from_filename(f.name)
            run_ts = parse_run_time_from_parent(f.parent.name)
        except Exception:
            continue

        # chỉ giữ timestamp trong đúng cửa sổ 7 ngày
        if not (START_TS <= valid_ts <= END_TS):
            continue

        # chỉ giữ file có run <= valid (forecast phải được phát hành trước hoặc đúng lúc)
        if run_ts > valid_ts:
            continue

        lead_hours = int((valid_ts - run_ts).total_seconds() // 3600)

        meta_rows.append({
            "file_path": str(f),
            "valid_ts": valid_ts,
            "run_ts": run_ts,
            "lead_hours": lead_hours,
        })

    meta_df = pd.DataFrame(meta_rows)

    if meta_df.empty:
        print("Khong co file SILAM hop le trong cua so 7 ngay")
        return

    print(f"So ban ghi hop le trong cua so 7 ngay: {len(meta_df)}")

    # Bước 2: mỗi valid_ts chỉ giữ forecast từ run gần nhất
    # Tức là run_ts lớn nhất, tương đương lead_hours nhỏ nhất
    meta_df = meta_df.sort_values(
        ["valid_ts", "run_ts", "lead_hours"],
        ascending=[True, False, True]
    )

    best_df = meta_df.drop_duplicates(subset=["valid_ts"], keep="first").copy()
    best_df = best_df.sort_values("valid_ts").reset_index(drop=True)

    print(f"So timestamp SILAM duy nhat sau khi chon run gan nhat: {len(best_df)}")
    print(best_df.head(10))

    rows = []

    for _, rec in best_df.iterrows():
        ts = rec["valid_ts"]
        file_path = Path(rec["file_path"])

        ds = Dataset(file_path)

        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        pm25 = ds.variables["cnc_PM2_5"][:]   # (1, lat, lon)
        pm10 = ds.variables["cnc_PM10"][:]    # (1, lat, lon)
        aqi = ds.variables["AQI"][:]          # (1, lat, lon)

        for _, st in stations.iterrows():
            lat_idx = find_nearest_idx(lats, st["lat"])
            lon_idx = find_nearest_idx(lons, st["lon"])

            rows.append({
                "timestamp": ts,
                "station_id": st["station_id"],
                "station_name": st["station_name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "silam_pm25": float(pm25[0, lat_idx, lon_idx]),
                "silam_pm10": float(pm10[0, lat_idx, lon_idx]),
                "silam_aqi": float(aqi[0, lat_idx, lon_idx]),
                "source_file": str(file_path),
                "run_time": rec["run_ts"],
                "lead_hours": rec["lead_hours"],
            })

        ds.close()

    out_df = pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("\n=== SILAM station series 7 ngay ===")
    print(out_df.head(12))
    print(f"\nShape: {out_df.shape}")
    print("\nStations:", out_df["station_id"].unique())
    print("\nDate range:", out_df["timestamp"].min(), "->", out_df["timestamp"].max())

    out_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
