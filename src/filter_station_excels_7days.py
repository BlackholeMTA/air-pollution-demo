import os
import glob
import pandas as pd

INPUT_DIR = "data/raw/stations"
OUTPUT_DIR = "data/processed/stations_filtered"

START_TIME = pd.Timestamp("2025-12-25 00:00:00")
END_TIME   = pd.Timestamp("2025-12-31 23:59:59")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_datetime_series(series):
    """
    Thử parse cột thời gian theo nhiều cách.
    """
    s = series.astype(str).str.strip()

    # thử parse dd/mm/yyyy có cả giờ
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # nếu vẫn lỗi nhiều thì thử lại kiểu thường
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(s, errors="coerce")

    return dt


def find_datetime_column(df):
    """
    Tìm cột có khả năng là cột thời gian bằng cách thử parse từng cột.
    """
    best_col = None
    best_count = 0

    for col in df.columns:
        parsed = parse_datetime_series(df[col])
        count_valid = parsed.notna().sum()
        if count_valid > best_count:
            best_count = count_valid
            best_col = col

    return best_col, best_count


def read_sheet_flexibly(path, sheet_name):
    """
    Đọc sheet theo nhiều kiểu header khác nhau để bắt đúng vùng dữ liệu thật.
    """
    candidates = []

    # thử các kiểu header khác nhau
    for header_row in [0, 1, 2, 3, 4, 5]:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
            candidates.append(df)
        except Exception:
            pass

    # thử không có header
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        candidates.append(df)
    except Exception:
        pass

    best_df = None
    best_col = None
    best_score = 0

    for df in candidates:
        if df is None or df.empty:
            continue

        # chuẩn hóa tên cột
        df.columns = [str(c).strip() for c in df.columns]

        col, score = find_datetime_column(df)
        if score > best_score:
            best_df = df.copy()
            best_col = col
            best_score = score

    return best_df, best_col, best_score


def filter_one_sheet(path, sheet_name):
    df, time_col, score = read_sheet_flexibly(path, sheet_name)

    if df is None or time_col is None or score == 0:
        return None, None, 0

    df = df.copy()
    df["__parsed_time__"] = parse_datetime_series(df[time_col])
    df = df.dropna(subset=["__parsed_time__"])

    filtered = df[(df["__parsed_time__"] >= START_TIME) & (df["__parsed_time__"] <= END_TIME)].copy()

    # đưa cột thời gian đã parse lên tên chuẩn
    filtered = filtered.rename(columns={"__parsed_time__": "timestamp_7days"})

    return filtered, time_col, score


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))

    if not files:
        print("Không tìm thấy file Excel nào trong", INPUT_DIR)
        return

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"\n==============================")
        print(f"ĐANG XỬ LÝ: {file_name}")
        print(f"==============================")

        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        except Exception as e:
            print(f"[LỖI] Không mở được file: {e}")
            continue

        output_path = os.path.join(OUTPUT_DIR, file_name)
        wrote_any_data = False

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet in sheet_names:
                try:
                    filtered_df, time_col, score = filter_one_sheet(file_path, sheet)

                    if filtered_df is not None and len(filtered_df) > 0:
                        filtered_df.to_excel(writer, sheet_name=sheet[:31], index=False)
                        wrote_any_data = True
                        print(
                            f"[OK] Sheet '{sheet}' | cột thời gian: '{time_col}' | "
                            f"parse được: {score} dòng | giữ lại: {len(filtered_df)} dòng"
                        )
                    else:
                        print(f"[BỎ QUA] Sheet '{sheet}' không tìm được dữ liệu trong 7 ngày")
                except Exception as e:
                    print(f"[LỖI] Sheet '{sheet}': {e}")

        if wrote_any_data:
            print(f"[XONG] Đã lưu: {output_path}")
        else:
            print(f"[CẢNH BÁO] File này không lọc ra được dữ liệu 7 ngày nào")


if __name__ == "__main__":
    main()
