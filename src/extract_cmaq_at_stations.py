from pathlib import Path
import pandas as pd

MAP_FILE = Path("data/processed/station_cmaq_mapping.csv")
CMAQ_FILE = Path("data/processed/cmaq_pm25_approx.csv")
OUT_FILE = Path("data/processed/cmaq_station_pm25.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    map_df = pd.read_csv(MAP_FILE)
    cmaq_df = pd.read_csv(CMAQ_FILE)

    merged = cmaq_df.merge(
        map_df[["station_id", "station_name", "row", "col"]],
        on=["row", "col"],
        how="inner"
    )

    merged = merged[[
        "time_idx", "station_id", "station_name", "row", "col", "pm25_cmaq_approx"
    ]].sort_values(["station_id", "time_idx"]).reset_index(drop=True)

    merged.to_csv(OUT_FILE, index=False)

    print(merged.head(20))
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
