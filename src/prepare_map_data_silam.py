from pathlib import Path
import pandas as pd

PRED_FILE = Path("data/output/predictions_silam_hanoi_full.csv")
META_FILE = Path("data/raw/stations/station_metadata.csv")
OUT_FILE = Path("data/output/map_data_silam_hanoi.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    pred_df = pd.read_csv(PRED_FILE, parse_dates=["timestamp"])
    meta_df = pd.read_csv(META_FILE)

    merged = pred_df.merge(
        meta_df,
        on=["station_id", "station_name"],
        how="left"
    )

    cols = [
        "timestamp",
        "station_id",
        "station_name",
        "lat",
        "lon",
        "pm25_obs",
        "silam_pm25",
        "pm25_pred_corrected"
    ]
    cols = [c for c in cols if c in merged.columns]

    merged = merged[cols].copy()
    merged.to_csv(OUT_FILE, index=False)

    print(merged.head(10))
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
