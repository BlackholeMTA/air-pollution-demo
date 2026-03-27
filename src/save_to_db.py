import os
import pandas as pd
import psycopg2
from pathlib import Path

PRED_PATH = Path("data/output/predictions.csv")

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "air_demo")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay file {PRED_PATH}")

    df = pd.read_csv(PRED_PATH)

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()

    insert_sql = """
    INSERT INTO pm25_predictions (
        timestamp, station_id, pm25_raw, pm25_obs, pm25_lag1, pm25_lag3,
        pm25_roll6, hour, dayofweek, target_pm25, pm25_pred_corrected
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cur.execute(insert_sql, (
            row.get("timestamp"),
            row.get("station_id"),
            float(row.get("pm25_raw")),
            float(row.get("pm25_obs")),
            float(row.get("pm25_lag1")),
            float(row.get("pm25_lag3")),
            float(row.get("pm25_roll6")),
            int(row.get("hour")),
            int(row.get("dayofweek")),
            float(row.get("target_pm25")),
            float(row.get("pm25_pred_corrected"))
        ))

    conn.commit()
    cur.close()
    conn.close()

    print(f"Da ghi {len(df)} dong vao database PostgreSQL")


if __name__ == "__main__":
    main()
