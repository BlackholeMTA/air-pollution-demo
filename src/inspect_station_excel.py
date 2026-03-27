import pandas as pd
from pathlib import Path

FILE_PATH = Path("data/raw/stations/3 tram HN 2025/AQI/Theo giờ/Hà Nội_ Công viên Nhân Chính - Khuất Duy Tiến (KK)_20260325_135207.xlsx")

def main():
    df = pd.read_excel(FILE_PATH)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print("\nHead:")
    print(df.head())

if __name__ == "__main__":
    main()
