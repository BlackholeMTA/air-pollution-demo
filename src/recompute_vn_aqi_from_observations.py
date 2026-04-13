import numpy as np
import pandas as pd

IN_PATH = "data/processed/final_training_table_with_features_7days.csv"
OUT_PATH = "data/output/vn_aqi_recomputed_from_observations.csv"

AQI_BREAKPOINTS = {
    "o3_1h": [(0, 0), (160, 50), (200, 100), (300, 150), (400, 200), (800, 300), (1000, 400), (1200, 500)],
    "co":    [(0, 0), (10000, 50), (30000, 100), (45000, 150), (60000, 200), (90000, 300), (120000, 400), (150000, 500)],
    "so2":   [(0, 0), (125, 50), (350, 100), (550, 150), (800, 200), (1600, 300), (2100, 400), (2630, 500)],
    "no2":   [(0, 0), (100, 50), (200, 100), (700, 150), (1200, 200), (2350, 300), (3100, 400), (3850, 500)],
    "pm10":  [(0, 0), (50, 50), (150, 100), (250, 150), (350, 200), (420, 300), (500, 400), (600, 500)],
    "pm25":  [(0, 0), (25, 50), (50, 100), (80, 150), (150, 200), (250, 300), (350, 400), (500, 500)],
}


def linear_aqi(c, bps):
    if pd.isna(c):
        return np.nan
    for i in range(len(bps) - 1):
        bp_lo, i_lo = bps[i]
        bp_hi, i_hi = bps[i + 1]
        if bp_lo <= c <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo
    if c > bps[-1][0]:
        return 500.0
    return np.nan


def nowcast_from_series(values):
    recent = pd.Series(values).dropna()
    if len(recent) < 2:
        return np.nan

    recent12 = pd.Series(values[-12:])
    # theo logic đang dùng trong pipeline hiện tại
    if recent12.tail(3).notna().sum() < 2:
        return np.nan

    cmin = recent12.min()
    cmax = recent12.max()

    if pd.isna(cmin) or pd.isna(cmax) or cmax == 0:
        return np.nan

    w_star = cmin / cmax
    w = max(w_star, 0.5)

    vals = []
    weights = []
    power = 0
    for v in recent12[::-1]:
        if pd.notna(v):
            vals.append(v)
            weights.append(w ** power)
        power += 1

    if len(vals) == 0 or sum(weights) == 0:
        return np.nan

    return float(np.dot(vals, weights) / sum(weights))


def classify_diff(diff):
    if pd.isna(diff):
        return "Không so sánh được"
    adiff = abs(diff)
    if adiff <= 10:
        return "Rất sát"
    elif adiff <= 25:
        return "Khá sát"
    elif adiff <= 50:
        return "Lệch vừa"
    else:
        return "Lệch nhiều"


def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["station_id", "timestamp"]).copy()

    results = []

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("timestamp").copy()

        pm25_nowcast_obs_list = []
        pm10_nowcast_obs_list = []

        for i in range(len(g)):
            pm25_nowcast_obs = nowcast_from_series(
                g["pm25_obs"].iloc[max(0, i - 11):i + 1].tolist()
            ) if "pm25_obs" in g.columns else np.nan

            pm10_nowcast_obs = nowcast_from_series(
                g["pm10_obs"].iloc[max(0, i - 11):i + 1].tolist()
            ) if "pm10_obs" in g.columns else np.nan

            pm25_nowcast_obs_list.append(pm25_nowcast_obs)
            pm10_nowcast_obs_list.append(pm10_nowcast_obs)

        g["pm25_nowcast_obs"] = pm25_nowcast_obs_list
        g["pm10_nowcast_obs"] = pm10_nowcast_obs_list

        for _, row in g.iterrows():
            aqi_pm25_obs = linear_aqi(row.get("pm25_nowcast_obs", np.nan), AQI_BREAKPOINTS["pm25"])
            aqi_pm10_obs = linear_aqi(row.get("pm10_nowcast_obs", np.nan), AQI_BREAKPOINTS["pm10"])
            aqi_o3_obs = linear_aqi(row.get("o3_obs", np.nan), AQI_BREAKPOINTS["o3_1h"])
            aqi_so2_obs = linear_aqi(row.get("so2_obs", np.nan), AQI_BREAKPOINTS["so2"])
            aqi_co_obs = linear_aqi(row.get("co_obs", np.nan), AQI_BREAKPOINTS["co"])
            aqi_no2_obs = linear_aqi(row.get("no2_obs", np.nan), AQI_BREAKPOINTS["no2"])

            aqi_parts = {
                "pm25": aqi_pm25_obs,
                "pm10": aqi_pm10_obs,
                "o3": aqi_o3_obs,
                "so2": aqi_so2_obs,
                "co": aqi_co_obs,
                "no2": aqi_no2_obs,
            }

            # giữ cùng logic với pipeline hiện tại:
            # phải có ít nhất 1 trong 2 thông số PM
            if pd.isna(aqi_pm25_obs) and pd.isna(aqi_pm10_obs):
                vn_aqi_recomputed = np.nan
                dominant_pollutant_recomputed = None
            else:
                valid_parts = {k: v for k, v in aqi_parts.items() if pd.notna(v)}
                if len(valid_parts) == 0:
                    vn_aqi_recomputed = np.nan
                    dominant_pollutant_recomputed = None
                else:
                    dominant_pollutant_recomputed = max(valid_parts, key=valid_parts.get)
                    vn_aqi_recomputed = round(valid_parts[dominant_pollutant_recomputed])

            vn_aqi_actual = row.get("vn_aqi", np.nan)
            diff = np.nan
            abs_diff = np.nan
            diff_group = "Không so sánh được"

            if pd.notna(vn_aqi_actual) and pd.notna(vn_aqi_recomputed):
                diff = float(vn_aqi_recomputed - vn_aqi_actual)
                abs_diff = abs(diff)
                diff_group = classify_diff(diff)

            results.append({
                "station_id": row.get("station_id"),
                "timestamp": row.get("timestamp"),
                "vn_aqi": vn_aqi_actual,
                "vn_aqi_recomputed_from_obs": vn_aqi_recomputed,
                "diff_recomputed_minus_actual": diff,
                "abs_diff": abs_diff,
                "diff_group": diff_group,
                "dominant_pollutant_recomputed": dominant_pollutant_recomputed,
                "pm25_nowcast_obs": row.get("pm25_nowcast_obs", np.nan),
                "pm10_nowcast_obs": row.get("pm10_nowcast_obs", np.nan),
                "aqi_pm25_obs": aqi_pm25_obs,
                "aqi_pm10_obs": aqi_pm10_obs,
                "aqi_o3_obs": aqi_o3_obs,
                "aqi_so2_obs": aqi_so2_obs,
                "aqi_co_obs": aqi_co_obs,
                "aqi_no2_obs": aqi_no2_obs,
                "pm25_obs": row.get("pm25_obs", np.nan),
                "pm10_obs": row.get("pm10_obs", np.nan),
                "o3_obs": row.get("o3_obs", np.nan),
                "so2_obs": row.get("so2_obs", np.nan),
                "co_obs": row.get("co_obs", np.nan),
                "no2_obs": row.get("no2_obs", np.nan),
            })

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("ĐÃ LƯU:", OUT_PATH)
    print("Số dòng:", len(out))

    compare = out.dropna(subset=["vn_aqi", "vn_aqi_recomputed_from_obs"]).copy()
    print("\nSố dòng so sánh được:", len(compare))

    if not compare.empty:
        mae = np.mean(np.abs(compare["vn_aqi_recomputed_from_obs"] - compare["vn_aqi"]))
        rmse = np.sqrt(np.mean((compare["vn_aqi_recomputed_from_obs"] - compare["vn_aqi"]) ** 2))
        rng = compare["vn_aqi"].max() - compare["vn_aqi"].min()
        nrmse = rmse / rng if rng != 0 else np.nan

        print("MAE =", round(mae, 3))
        print("RMSE =", round(rmse, 3))
        print("NRMSE =", round(nrmse, 3) if pd.notna(nrmse) else np.nan)

        print("\nPhân nhóm độ lệch:")
        print(compare["diff_group"].value_counts(dropna=False))

        print("\nMột số dòng đầu để kiểm tra:")
        print(compare[[
            "timestamp", "station_id", "vn_aqi", "vn_aqi_recomputed_from_obs",
            "diff_recomputed_minus_actual", "dominant_pollutant_recomputed"
        ]].head(20))
    else:
        print("Không có dòng nào đủ để so sánh.")


if __name__ == "__main__":
    main()
