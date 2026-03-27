from pathlib import Path
import pandas as pd

STATION_FILE = Path("data/processed/hanoi_hourly_merged.csv")
SILAM_FILE = Path("data/processed/silam_station_pm25_2025122700.csv")
OUT_FILE = Path("data/processed/train_silam_hanoi_2025122700.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    station_df = pd.read_csv(STATION_FILE, parse_dates=["timestamp"])
    silam_df = pd.read_csv(SILAM_FILE)

    sub = station_df[
        station_df["timestamp"].astype(str) == "2025-12-27 00:00:00"
    ].copy()

    merged = sub.merge(
        silam_df[["station_id", "station_name", "silam_pm25", "silam_pm10", "silam_aqi"]],
        on=["station_id", "station_name"],
        how="inner"
    )

    print(merged)
    print(f"\nShape: {merged.shape}")

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
