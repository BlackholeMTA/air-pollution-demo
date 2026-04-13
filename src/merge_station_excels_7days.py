import os
import glob
import pandas as pd

INPUT_DIR = "data/processed/stations_filtered"
OUTPUT_DIR = "data/processed/stations_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nhận diện trạm từ tên file
STATION_PATTERNS = {
    "nguyen_van_cu": ["556 Nguyễn Văn Cừ"],
    "nhan_chinh": ["Công viên Nhân Chính", "Khuất Duy Tiến"],
    "giai_phong": ["Số 1 đường Giải Phóng", "Bạch Mai"]
}

# Chuẩn hóa tên cột
COLUMN_RENAME = {
    "Datetime": "datetime_raw",
    "timestamp_7days": "timestamp",
    "VN_AQI": "vn_aqi",
    "PM-10": "pm10_obs",
    "PM10": "pm10_obs",
    "PM-2-5": "pm25_obs",
    "PM2.5": "pm25_obs",
    "PM_2.5": "pm25_obs",
    "O3": "o3_obs",
    "SO2": "so2_obs",
    "CO": "co_obs",
    "NO2": "no2_obs",
    "NOx": "nox_obs",
    "NO": "no_obs",
    "RH": "rh",
    "Độ ẩm": "rh",
    "Do am": "rh",
    "Nhiệt độ": "temp",
    "Nhiet do": "temp",
    "Tốc độ gió": "wind_speed",
    "Toc do gio": "wind_speed",
    "Hướng gió": "wind_dir",
    "Huong gio": "wind_dir",
    "Áp suất": "pressure",
    "Ap suat": "pressure",
    "Bức xạ": "radiation",
    "Buc xa": "radiation",
}

def detect_station_id(filename: str):
    for station_id, keywords in STATION_PATTERNS.items():
        for kw in keywords:
            if kw.lower() in filename.lower():
                return station_id
    return None

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c in COLUMN_RENAME:
            rename_map[c] = COLUMN_RENAME[c]
    df = df.rename(columns=rename_map)
    return df

def pick_best_sheet(path):
    xls = pd.ExcelFile(path)
    best_df = None
    best_rows = -1
    best_sheet = None

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
            if len(df) > best_rows:
                best_rows = len(df)
                best_df = df
                best_sheet = sheet
        except Exception:
            continue

    return best_df, best_sheet

def prepare_df(df):
    df = normalize_columns(df)

    if "timestamp" not in df.columns:
        raise ValueError("Không có cột timestamp sau khi chuẩn hóa.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # bỏ cột rỗng hoàn toàn
    df = df.dropna(axis=1, how="all")

    # bỏ trùng timestamp nếu có
    df = df.sort_values("timestamp")
    df = df.groupby("timestamp", as_index=False).first()

    return df

def merge_station_files(file_list, station_id):
    prepared = []

    for f in file_list:
        df, sheet = pick_best_sheet(f)
        if df is None:
            print(f"[BỎ QUA] Không đọc được file {f}")
            continue
        try:
            df = prepare_df(df)
            prepared.append((os.path.basename(f), df))
            print(f"[OK] {os.path.basename(f)} | sheet={sheet} | rows={len(df)}")
        except Exception as e:
            print(f"[LỖI] {os.path.basename(f)}: {e}")

    if not prepared:
        return None

    merged = prepared[0][1].copy()

    for fname, df in prepared[1:]:
        # tránh đè cột trùng
        dup_cols = [c for c in df.columns if c in merged.columns and c != "timestamp"]
        if dup_cols:
            df = df.drop(columns=dup_cols)
        merged = pd.merge(merged, df, on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["station_id"] = station_id

    # nếu chưa có nox_obs mà có no_obs + no2_obs thì tự cộng
    if "nox_obs" not in merged.columns and "no_obs" in merged.columns and "no2_obs" in merged.columns:
        merged["nox_obs"] = merged["no_obs"].fillna(0) + merged["no2_obs"].fillna(0)

    return merged

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))
    station_groups = {}

    for f in files:
        station_id = detect_station_id(os.path.basename(f))
        if station_id is None:
            print(f"[BỎ QUA] Không nhận diện được trạm từ file: {os.path.basename(f)}")
            continue
        station_groups.setdefault(station_id, []).append(f)

    for station_id, group_files in station_groups.items():
        print("\n==============================")
        print(f"ĐANG GỘP TRẠM: {station_id}")
        print("==============================")

        merged = merge_station_files(group_files, station_id)
        if merged is None or merged.empty:
            print(f"[CẢNH BÁO] Không gộp được dữ liệu cho trạm {station_id}")
            continue

        out_xlsx = os.path.join(OUTPUT_DIR, f"{station_id}_merged_7days.xlsx")
        out_csv = os.path.join(OUTPUT_DIR, f"{station_id}_merged_7days.csv")

        merged.to_excel(out_xlsx, index=False)
        merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"[XONG] Đã lưu:")
        print(f"  - {out_xlsx}")
        print(f"  - {out_csv}")
        print("Cột:", merged.columns.tolist())
        print("Số dòng:", len(merged))

if __name__ == "__main__":
    main()
