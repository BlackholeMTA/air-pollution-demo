from netCDF4 import Dataset
from pathlib import Path

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")

KEYS = [
    "GDTYP", "P_ALP", "P_BET", "P_GAM",
    "XCENT", "YCENT", "XORIG", "YORIG",
    "XCELL", "YCELL", "NCOLS", "NROWS", "NLAYS"
]

def main():
    ds = Dataset(NC_FILE)

    print("=== PROJECTION ATTRIBUTES ===")
    for k in KEYS:
        if hasattr(ds, k):
            print(f"{k} = {getattr(ds, k)}")
        else:
            print(f"{k} = NOT_FOUND")

    ds.close()

if __name__ == "__main__":
    main()
