from pathlib import Path
import pandas as pd

BASE = Path("data/raw/stations/3 tram HN 2025")

AQI_DIR = BASE / "AQI" / "Theo giờ"
OTH_DIR = BASE / "Thông số khác" / "Theo giờ"

OUT_FILE = Path("data/processed/hanoi_hourly_merged.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def clean_col_name(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", " ")
    col = col.replace("  ", " ")
    return col


def normalize_datetime(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # Ưu tiên parse dạng ngày/tháng/năm giờ:phút
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt


def read_excel_clean(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [clean_col_name(c) for c in df.columns]
    return df


def parse_station_from_name(filename: str):
    name = filename.lower()

    if "nguyễn văn cừ" in name:
        return "HN_NVC", "556 Nguyễn Văn Cừ"
    if "nhân chính" in name or "khuất duy tiến" in name:
        return "HN_NC", "Công viên Nhân Chính - Khuất Duy Tiến"
    if "giải phóng" in name or "bạch mai" in name:
        return "HN_GP", "Số 1 Giải Phóng - Bạch Mai"

    return "UNKNOWN", filename


def load_aqi_files() -> pd.DataFrame:
    frames = []

    for path in AQI_DIR.glob("*.xlsx"):
        if ":Zone.Identifier" in path.name:
            continue

        station_id, station_name = parse_station_from_name(path.name)
        df = read_excel_clean(path)

        # Chuẩn hóa tên cột theo ảnh bạn gửi
        rename_map = {
            "Datetime": "timestamp",
            "VN_AQI": "vn_aqi",
            "VN_AQI ": "vn_aqi",
            "PM-10": "pm10_aqi",
            "PM-2-5": "pm25_aqi",
        }
        df = df.rename(columns=rename_map)

        needed = ["timestamp", "vn_aqi", "pm10_aqi", "pm25_aqi"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

        df["timestamp"] = normalize_datetime(df["timestamp"])
        df["station_id"] = station_id
        df["station_name"] = station_name

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out


def load_other_files() -> pd.DataFrame:
    frames = []

    for path in OTH_DIR.glob("*.xlsx"):
        if ":Zone.Identifier" in path.name:
            continue

        station_id, station_name = parse_station_from_name(path.name)
        df = read_excel_clean(path)

        # Chuẩn hóa tên cột từ bộ "Thông số khác"
        rename_map = {
            "Datetime": "timestamp",
            "SO2": "so2",
            "O3": "o3",
            "NO2": "no2",
            "CO": "co",
            "PM-10": "pm10_obs",
            "PM-2-5": "pm25_obs",
            "PM-10(µg/Nm3}": "pm10_obs",
            "PM-2-5(µg/Nm3}": "pm25_obs",
            "PM-10(µg/Nm3)": "pm10_obs",
            "PM-2-5(µg/Nm3)": "pm25_obs",
        }
        df = df.rename(columns=rename_map)

        keep_candidates = [
            "timestamp", "so2", "o3", "no2", "co", "pm10_obs", "pm25_obs"
        ]
        existing = [c for c in keep_candidates if c in df.columns]
        df = df[existing].copy()

        # thay dấu "-" thành NaN
        for c in df.columns:
            if c != "timestamp":
                df[c] = df[c].replace("-", pd.NA)

        df["timestamp"] = normalize_datetime(df["timestamp"])
        df["station_id"] = station_id
        df["station_name"] = station_name

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out


def main():
    aqi_df = load_aqi_files()
    oth_df = load_other_files()

    print("AQI shape:", aqi_df.shape)
    print("OTH shape:", oth_df.shape)

    merged = pd.merge(
        aqi_df,
        oth_df,
        on=["timestamp", "station_id", "station_name"],
        how="outer"
    )

    merged = merged.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("\nMerged shape:", merged.shape)
    print("\nColumns:", merged.columns.tolist())
    print("\nHead:")
    print(merged.head(10))

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
