import os
from netCDF4 import Dataset

FILES_TO_CHECK = [
    "data/raw/cmaq/20251225/CCTM_ACONC_v532_gcc_v53_20251225.nc",
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

    print("\n=== KIỂM TRA CÁC BIẾN CHÍNH ===")
    for v in ["NO2", "NO", "O3", "SO2", "CO", "RH", "TA", "PRES", "WVEL", "LAT", "LON", "LATITUDE", "LONGITUDE"]:
        if v in ds.variables:
            var = ds.variables[v]
            print(f"{v} FOUND | dims={var.dimensions} | shape={var.shape}")

    ds.close()
