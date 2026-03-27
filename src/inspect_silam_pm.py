from netCDF4 import Dataset
from pathlib import Path

NC_FILE = Path("data/raw/silam/20251227/PM_2025122700.nc4")

def main():
    ds = Dataset(NC_FILE)

    print("=== GLOBAL ATTRIBUTES ===")
    for attr in ds.ncattrs():
        print(f"{attr}: {getattr(ds, attr)}")

    print("\n=== DIMENSIONS ===")
    for name, dim in ds.dimensions.items():
        print(f"{name}: {len(dim)}")

    print("\n=== VARIABLES ===")
    for name, var in ds.variables.items():
        print(f"{name} | dims={var.dimensions} | shape={var.shape}")

    ds.close()

if __name__ == "__main__":
    main()
