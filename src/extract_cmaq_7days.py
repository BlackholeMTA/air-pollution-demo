import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset

BASE_DIR = "data/raw/cmaq"
GRID_MAP_PATH = "data/raw/cmaq/station_cmaq_grid.csv"
OUT_PATH = "data/processed/cmaq_7days_all_stations.csv"

DATES = ["20251225", "20251226", "20251227", "20251228", "20251229", "20251230", "20251231"]

CMAQ_VARS = {
    "cmaq_o3": "O3",
    "cmaq_so2": "SO2",
    "cmaq_co": "CO",
    "cmaq_no2": "NO2",
    "cmaq_no": "NO",
    "cmaq_rh": "RH",
    "cmaq_ta": "TA",
    "cmaq_pres": "PRES",
    "cmaq_wvel": "WVEL",
}


def read_grid_map():
    df = pd.read_csv(GRID_MAP_PATH)
    required = {"station_id", "row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"station_cmaq_grid.csv thiếu cột: {missing}")
    return df.copy()


def parse_tflag_to_timestamp(ds):
    """
    TFLAG shape = (TSTEP, VAR, DATE-TIME)
    DATE-TIME:
      [:, :, 0] = YYYYDDD
      [:, :, 1] = HHMMSS
    Lấy theo biến đầu tiên VAR=0 là đủ.
    """
    if "TFLAG" not in ds.variables:
        return None

    tflag = ds.variables["TFLAG"][:]
    timestamps = []

    for t in range(tflag.shape[0]):
        yyyyddd = int(tflag[t, 0, 0])
        hhmmss = int(tflag[t, 0, 1])

        year = yyyyddd // 1000
        ddd = yyyyddd % 1000

        hh = hhmmss // 10000
        mm = (hhmmss % 10000) // 100
        ss = hhmmss % 100

        ts = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=ddd - 1, hours=hh, minutes=mm, seconds=ss)
        timestamps.append(ts)

    return timestamps


def extract_one_day(day, grid_map):
    nc_path = os.path.join(BASE_DIR, day, f"CCTM_ACONC_v532_gcc_v53_{day}.nc")

    if not os.path.exists(nc_path):
        print(f"[WARN] Thiếu file CMAQ ngày {day}: {nc_path}")
        return pd.DataFrame()

    ds = Dataset(nc_path)

    timestamps = parse_tflag_to_timestamp(ds)
    if timestamps is None:
        # fallback: giả sử 24 giờ trong ngày
        timestamps = pd.date_range(start=pd.to_datetime(day, format="%Y%m%d"), periods=24, freq="H")

    rows = []

    for _, st in grid_map.iterrows():
        station_id = st["station_id"]
        row_idx = int(st["row"])
        col_idx = int(st["col"])

        for t_idx, ts in enumerate(timestamps):
            row = {
                "timestamp": ts,
                "station_id": station_id,
                "cmaq_row": row_idx,
                "cmaq_col": col_idx,
                "source_file": os.path.basename(nc_path),
            }

            for out_col, var_name in CMAQ_VARS.items():
                if var_name in ds.variables:
                    try:
                        row[out_col] = float(ds.variables[var_name][t_idx, 0, row_idx, col_idx])
                    except Exception:
                        row[out_col] = np.nan
                else:
                    row[out_col] = np.nan

            rows.append(row)

    ds.close()

    df = pd.DataFrame(rows)

    if "cmaq_no" in df.columns and "cmaq_no2" in df.columns:
        df["cmaq_nox"] = df["cmaq_no"].fillna(0) + df["cmaq_no2"].fillna(0)

    return df


def main():
    grid_map = read_grid_map()
    frames = []

    for day in DATES:
        print(f"\nĐANG XỬ LÝ NGÀY: {day}")
        df_day = extract_one_day(day, grid_map)
        print("  rows =", len(df_day))
        frames.append(df_day)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\nĐÃ LƯU:", OUT_PATH)
    print("Số dòng:", len(df))
    print("Cột:", df.columns.tolist())

    if not df.empty:
        print("\nĐộ phủ theo trạm:")
        for station_id, g in df.groupby("station_id"):
            print(f"\n== {station_id} ==")
            print("rows =", len(g))
            for c in ["cmaq_o3", "cmaq_so2", "cmaq_co", "cmaq_no2", "cmaq_no", "cmaq_nox", "cmaq_rh", "cmaq_ta", "cmaq_pres", "cmaq_wvel"]:
                if c in g.columns:
                    print(f"{c}: {g[c].notna().sum()}")


if __name__ == "__main__":
    main()
