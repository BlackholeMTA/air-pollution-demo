import os
import glob

BASE_DIR = "data/raw/silam"
DATES = ["20251225", "20251226", "20251227", "20251228", "20251229", "20251230", "20251231"]

for d in DATES:
    day_dir = os.path.join(BASE_DIR, d)
    print("\n" + "=" * 80)
    print("NGÀY:", d)
    print("THƯ MỤC:", day_dir)
    print("=" * 80)

    if not os.path.exists(day_dir):
        print("KHÔNG TỒN TẠI")
        continue

    gas_files = sorted(glob.glob(os.path.join(day_dir, f"{d}00.nc4")))
    pm_files = sorted(glob.glob(os.path.join(day_dir, "PM_*.nc4")))
    aqi_files = sorted(glob.glob(os.path.join(day_dir, "AQI*.nc4")))

    print("Gas files:", len(gas_files))
    for f in gas_files[:5]:
        print("  ", os.path.basename(f))

    print("PM files:", len(pm_files))
    for f in pm_files[:10]:
        print("  ", os.path.basename(f))

    print("AQI files:", len(aqi_files))
    for f in aqi_files[:10]:
        print("  ", os.path.basename(f))
