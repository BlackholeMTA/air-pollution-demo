from pathlib import Path

BASE = Path("data/raw/stations/3 tram HN 2025")

def main():
    for sub in [
        "AQI/Theo giờ",
        "AQI/Theo ngày",
        "Thông số khác/Theo giờ",
        "Thông số khác/Theo ngày",
    ]:
        folder = BASE / sub
        print(f"\n=== {folder} ===")
        for f in folder.iterdir():
            if f.is_file() and ":Zone.Identifier" not in f.name:
                print(f.name)

if __name__ == "__main__":
    main()
