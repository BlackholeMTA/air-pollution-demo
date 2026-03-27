from netCDF4 import Dataset
from pathlib import Path

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")

KEYWORDS = ["LAT", "LON", "X", "Y", "XCELL", "YCELL", "XORIG", "YORIG"]

def main():
    ds = Dataset(NC_FILE)

    print("=== GLOBAL ATTRIBUTES ===")
    for attr in ds.ncattrs():
        val = getattr(ds, attr)
        if any(k in attr.upper() for k in KEYWORDS):
            print(attr, "=", val)

    print("\n=== VARIABLES RELATED TO GEO ===")
    for name, var in ds.variables.items():
        if any(k in name.upper() for k in KEYWORDS):
            print(f"{name} | dims={var.dimensions} | shape={var.shape}")

    ds.close()

if __name__ == "__main__":
    main()
