import numpy as np
import pandas as pd

IN_PATH = "data/output/predictions_6pollutants_7days_gbtree.csv"
OUT_PATH = "data/output/vn_aqi_from_predictions_7days_gbtree.csv"

AQI_BREAKPOINTS = {
    "o3_1h": [(0, 0), (160, 50), (200, 100), (300, 150), (400, 200), (800, 300), (1000, 400), (1200, 500)],
    "co":    [(0, 0), (10000, 50), (30000, 100), (45000, 150), (60000, 200), (90000, 300), (120000, 400), (150000, 500)],
    "so2":   [(0, 0), (125, 50), (350, 100), (550, 150), (800, 200), (1600, 300), (2100, 400), (2630, 500)],
    "no2":   [(0, 0), (100, 50), (200, 100), (700, 150), (1200, 200), (2350, 300), (3100, 400), (3850, 500)],
    "pm10":  [(0, 0), (50, 50), (150, 100), (250, 150), (350, 200), (420, 300), (500, 400), (600, 500)],
    "pm25":  [(0, 0), (25, 50), (50, 100), (80, 150), (150, 200), (250, 300), (350, 400), (500, 500)],
}

AQI_LEVELS = [
    (0, 50, "Tốt", "Xanh"),
    (51, 100, "Trung bình", "Vàng"),
    (101, 150, "Kém", "Da cam"),
    (151, 200, "Xấu", "Đỏ"),
    (201, 300, "Rất xấu", "Tím"),
    (301, 500, "Nguy hại", "Nâu"),
]

HEALTH_TEXT = {
    "Tốt": "Chất lượng không khí tốt, không ảnh hưởng tới sức khỏe.",
    "Trung bình": "Chất lượng không khí ở mức chấp nhận được; người nhạy cảm có thể chịu tác động nhất định.",
    "Kém": "Người nhạy cảm có thể gặp vấn đề sức khỏe, người bình thường ít bị ảnh hưởng.",
    "Xấu": "Người bình thường bắt đầu bị ảnh hưởng; người nhạy cảm có thể bị ảnh hưởng nghiêm trọng hơn.",
    "Rất xấu": "Mọi người đều có thể bị ảnh hưởng nghiêm trọng hơn tới sức khỏe.",
    "Nguy hại": "Cảnh báo khẩn cấp về sức khỏe; toàn bộ dân số có thể bị ảnh hưởng nghiêm trọng."
}

RECOMMEND_TEXT = {
    "Tốt": "Tự do hoạt động ngoài trời.",
    "Trung bình": "Tiếp tục hoạt động bình thường; nhóm nhạy cảm nên theo dõi triệu chứng.",
    "Kém": "Giảm hoạt động mạnh ngoài trời, đặc biệt với nhóm nhạy cảm.",
    "Xấu": "Hạn chế hoạt động ngoài trời; nhóm nhạy cảm nên ở trong nhà nhiều hơn.",
    "Rất xấu": "Hạn chế tối đa hoạt động ngoài trời; nếu phải ra ngoài hãy đeo khẩu trang đạt chuẩn.",
    "Nguy hại": "Nên ở trong nhà, đóng cửa và chỉ ra ngoài khi thật cần thiết."
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

def classify_aqi(aqi):
    if pd.isna(aqi):
        return None, None
    aqi = int(round(aqi))
    for lo, hi, level, color in AQI_LEVELS:
        if lo <= aqi <= hi:
            return level, color
    return "Nguy hại", "Nâu"

def compute_hourly_aqi(df):
    df = df.sort_values(["station_id", "timestamp"]).copy()
    result_rows = []

    for station_id, g in df.groupby("station_id"):
        g = g.sort_values("timestamp").copy()

        pm25_nowcast_list = []
        pm10_nowcast_list = []

        for i in range(len(g)):
            pm25_nowcast = nowcast_from_series(g["pm25_pred"].iloc[max(0, i-11):i+1].tolist()) if "pm25_pred" in g.columns else np.nan
            pm10_nowcast = nowcast_from_series(g["pm10_pred"].iloc[max(0, i-11):i+1].tolist()) if "pm10_pred" in g.columns else np.nan
            pm25_nowcast_list.append(pm25_nowcast)
            pm10_nowcast_list.append(pm10_nowcast)

        g["pm25_nowcast"] = pm25_nowcast_list
        g["pm10_nowcast"] = pm10_nowcast_list

        for _, row in g.iterrows():
            aqi_parts = {}
            aqi_parts["pm25"] = linear_aqi(row.get("pm25_nowcast", np.nan), AQI_BREAKPOINTS["pm25"])
            aqi_parts["pm10"] = linear_aqi(row.get("pm10_nowcast", np.nan), AQI_BREAKPOINTS["pm10"])
            aqi_parts["o3"] = linear_aqi(row.get("o3_pred", np.nan), AQI_BREAKPOINTS["o3_1h"])
            aqi_parts["so2"] = linear_aqi(row.get("so2_pred", np.nan), AQI_BREAKPOINTS["so2"])
            aqi_parts["co"] = linear_aqi(row.get("co_pred", np.nan), AQI_BREAKPOINTS["co"])
            aqi_parts["no2"] = linear_aqi(row.get("no2_pred", np.nan), AQI_BREAKPOINTS["no2"])

            if pd.isna(aqi_parts["pm25"]) and pd.isna(aqi_parts["pm10"]):
                vn_aqi = np.nan
                dominant = None
            else:
                valid_parts = {k: v for k, v in aqi_parts.items() if pd.notna(v)}
                if len(valid_parts) == 0:
                    vn_aqi = np.nan
                    dominant = None
                else:
                    dominant = max(valid_parts, key=valid_parts.get)
                    vn_aqi = round(valid_parts[dominant])

            level, color = classify_aqi(vn_aqi)

            result_rows.append({
                "station_id": station_id,
                "timestamp": row["timestamp"],
                "pm25_nowcast": row.get("pm25_nowcast", np.nan),
                "pm10_nowcast": row.get("pm10_nowcast", np.nan),
                "aqi_pm25": aqi_parts["pm25"],
                "aqi_pm10": aqi_parts["pm10"],
                "aqi_o3": aqi_parts["o3"],
                "aqi_so2": aqi_parts["so2"],
                "aqi_co": aqi_parts["co"],
                "aqi_no2": aqi_parts["no2"],
                "vn_aqi_hour": vn_aqi,
                "dominant_pollutant": dominant,
                "level": level,
                "color": color,
                "health_text": HEALTH_TEXT.get(level),
                "recommendation": RECOMMEND_TEXT.get(level),
            })

    return pd.DataFrame(result_rows)

def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    aqi_df = compute_hourly_aqi(df)
    aqi_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("ĐÃ LƯU:", OUT_PATH)
    print("Số dòng:", len(aqi_df))
    print("Độ phủ VN_AQI:", aqi_df["vn_aqi_hour"].notna().sum())
    print(aqi_df["level"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
