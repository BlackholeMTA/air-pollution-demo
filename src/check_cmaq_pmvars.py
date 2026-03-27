from netCDF4 import Dataset
from pathlib import Path

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")

KEYWORDS = [
    "PM25", "PM10", "ATOT", "PM25AT", "PM25AC", "PM25CO",
    "AER", "DIAG", "AERO", "ATOTI", "ATOTJ", "ATOTK"
]

def main():
    ds = Dataset(NC_FILE)

    print("=== VARIABLES LIÊN QUAN PM ===")
    for name, var in ds.variables.items():
        name_up = name.upper()
        if any(k in name_up for k in KEYWORDS):
            print(f"{name} | dims={var.dimensions} | shape={var.shape}")

    ds.close()

if __name__ == "__main__":
    main()
