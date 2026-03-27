from pathlib import Path
import pandas as pd

STATION_FILE = Path("data/processed/hanoi_hourly_merged.csv")
SILAM_FILE = Path("data/processed/silam_station_series.csv")
OUT_FILE = Path("data/processed/train_silam_hanoi_series.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    station_df = pd.read_csv(STATION_FILE, parse_dates=["timestamp"])
    silam_df = pd.read_csv(SILAM_FILE, parse_dates=["timestamp"])

    merged = station_df.merge(
        silam_df[["timestamp", "station_id", "station_name", "silam_pm25", "silam_pm10", "silam_aqi"]],
        on=["timestamp", "station_id", "station_name"],
        how="inner"
    )

    merged = merged.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print(merged.head(15))
    print(f"\nShape: {merged.shape}")
    print("\nStations:", merged["station_id"].unique())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
