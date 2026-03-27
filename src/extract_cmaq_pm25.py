from netCDF4 import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")
OUT_FILE = Path("data/processed/cmaq_pm25_approx.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Các species dùng để xấp xỉ PM2.5
PM25_SPECIES = [
    "ASO4I", "ASO4J",
    "ANH4I", "ANH4J",
    "ANO3I", "ANO3J",
    "AECI", "AECJ",
]

def main():
    ds = Dataset(NC_FILE)

    # Kiểm tra species có đủ không
    missing = [v for v in PM25_SPECIES if v not in ds.variables]
    if missing:
        print("Thiếu các biến:", missing)
        return

    tstep = len(ds.dimensions["TSTEP"])
    row_n = len(ds.dimensions["ROW"])
    col_n = len(ds.dimensions["COL"])

    rows = []

    for t in range(tstep):
        # cộng các species để ra PM2.5 xấp xỉ
        pm25 = np.zeros((row_n, col_n), dtype=float)

        for sp in PM25_SPECIES:
            pm25 += ds.variables[sp][t, 0, :, :]

        # lưu từng ô lưới
        for r in range(row_n):
            for c in range(col_n):
                rows.append({
                    "time_idx": t,
                    "row": r,
                    "col": c,
                    "pm25_cmaq_approx": float(pm25[r, c])
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"Saved to {OUT_FILE}")
    print(out_df.head())

    ds.close()

if __name__ == "__main__":
    main()
