from pathlib import Path
import pandas as pd

STATION_FILE = Path("data/processed/hanoi_hourly_merged.csv")
CMAQ_FILE = Path("data/processed/cmaq_station_pm25.csv")
OUT_FILE = Path("data/processed/train_cmaq_hanoi.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET_DATE = "2025-12-25"

def main():
    station_df = pd.read_csv(STATION_FILE, parse_dates=["timestamp"])
    cmaq_df = pd.read_csv(CMAQ_FILE)

    # lấy đúng ngày của file CMAQ
    station_day = station_df[
        station_df["timestamp"].dt.date.astype(str) == TARGET_DATE
    ].copy()

    # tạo time_idx từ giờ
    station_day["time_idx"] = station_day["timestamp"].dt.hour

    merged = station_day.merge(
        cmaq_df,
        on=["station_id", "station_name", "time_idx"],
        how="inner"
    )

    merged = merged.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("Merged shape:", merged.shape)
    print("\nColumns:", merged.columns.tolist())
    print("\nHead:")
    print(merged.head(10))

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == "__main__":
    main()
