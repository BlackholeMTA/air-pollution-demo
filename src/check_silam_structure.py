import os
from netCDF4 import Dataset

FILES_TO_CHECK = [
    "data/raw/silam/20251225/2025122500.nc4",
    "data/raw/silam/20251225/PM_2025122500.nc4",
    "data/raw/silam/20251225/AQI20251225.nc4",
]

for path in FILES_TO_CHECK:
    print("\n" + "=" * 80)
    print("FILE:", path)
    print("=" * 80)

    if not os.path.exists(path):
        print("KHÔNG TỒN TẠI")
        continue

    ds = Dataset(path)

    print("\n=== DIMENSIONS ===")
    for d in ds.dimensions:
        print(d, len(ds.dimensions[d]))

    print("\n=== VARIABLES ===")
    for v in ds.variables:
        var = ds.variables[v]
        print(f"{v} | dims={var.dimensions} | shape={var.shape}")

    ds.close()
