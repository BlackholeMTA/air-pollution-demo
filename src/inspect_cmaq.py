from netCDF4 import Dataset
from pathlib import Path

NC_FILE = Path("data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc")

def main():
    ds = Dataset(NC_FILE)

    print("=== DIMENSIONS ===")
    for name, dim in ds.dimensions.items():
        print(f"{name}: {len(dim)}")

    print("\n=== VARIABLES CONTAINING PM / AERO / ASO / ANO / AEC / NH4 / SO4 / NO3 ===")
    keywords = ["PM", "pm", "AERO", "ASO", "ANO", "AEC", "ANH4", "ASO4", "ANO3", "NH4", "SO4", "NO3"]

    for name, var in ds.variables.items():
        if any(k in name for k in keywords):
            print(f"{name} | dims={var.dimensions} | shape={var.shape}")

    print("\n=== ALL VARIABLES STARTING WITH A ===")
    for name, var in ds.variables.items():
        if name.startswith("A"):
            print(f"{name} | dims={var.dimensions} | shape={var.shape}")

    ds.close()

if __name__ == "__main__":
    main()
