from pathlib import Path
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import re

CMAQ_ROOT = Path("data/raw/cmaq")
MAP_FILE = Path("data/processed/station_cmaq_mapping.csv")
OUT_FILE = Path("data/processed/cmaq_station_series_7days.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

PM25_SPECIES = [
    "ASO4I", "ASO4J",
    "ANH4I", "ANH4J",
    "ANO3I", "ANO3J",
    "AECI", "AECJ",
]

def parse_date_from_filename(filename: str):
    # CCTM_ACONC_v532_gcc_v53_20251225.nc -> 2025-12-25
    m = re.search(r"(\d{8})", filename)
    if not m:
        raise ValueError(f"Khong parse duoc ngay tu ten file: {filename}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").date()

def main():
    map_df = pd.read_csv(MAP_FILE)

    files = sorted([
        f for f in CMAQ_ROOT.rglob("CCTM_ACONC*.nc")
        if ":Zone.Identifier" not in f.name
    ])

    print(f"So file CMAQ tim thay: {len(files)}")
    if not files:
        print("Khong tim thay file CCTM_ACONC*.nc")
        return

    rows = []

    for file_path in files:
        day = parse_date_from_filename(file_path.name)
        ds = Dataset(file_path)

        missing = [v for v in PM25_SPECIES if v not in ds.variables]
        if missing:
            print(f"Bo qua {file_path.name}, thieu bien: {missing}")
            ds.close()
            continue

        tstep = len(ds.dimensions["TSTEP"])

        for t in range(tstep):
            ts = pd.Timestamp(day) + pd.Timedelta(hours=t)

            pm25_grid = None
            for sp in PM25_SPECIES:
                arr = ds.variables[sp][t, 0, :, :]
                pm25_grid = arr if pm25_grid is None else pm25_grid + arr

            for _, st in map_df.iterrows():
                r = int(st["row"])
                c = int(st["col"])
                rows.append({
                    "timestamp": ts,
                    "station_id": st["station_id"],
                    "station_name": st["station_name"],
                    "row": r,
                    "col": c,
                    "cmaq_pm25_approx": float(pm25_grid[r, c]),
                })

        ds.close()

    out_df = pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    print(out_df.head(12))
    print(f"\nShape: {out_df.shape}")
    print("\nStations:", out_df["station_id"].unique())

    out_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
