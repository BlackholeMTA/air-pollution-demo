from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Dashboard chất lượng không khí Hà Nội",
    page_icon="🌫️",
    layout="wide"
)

# =========================
# CONFIG
# =========================
MODEL_CONFIG = {
    "HYBRID": {
        "pred_test": Path("data/output/predictions_multi_source_7days.csv"),
        "pred_full": Path("data/output/predictions_multi_source_7days_full.csv"),
        "plot": Path("data/output/comparison_multi_source_7days.png"),
        "map_data": Path("data/output/map_data_hybrid_7days.csv"),
        "raw_cols": ["silam_pm25", "cmaq_pm25_approx"],
        "default": True,
    },
    "SILAM": {
        "pred_test": Path("data/output/predictions_silam_hanoi.csv"),
        "pred_full": Path("data/output/predictions_silam_hanoi_full.csv"),
        "plot": Path("data/output/comparison_silam_hanoi.png"),
        "map_data": Path("data/output/map_data_silam_hanoi.csv"),
        "raw_cols": ["silam_pm25"],
        "default": False,
    },
    "CMAQ": {
        "pred_test": Path("data/output/predictions_cmaq_hanoi.csv"),
        "pred_full": Path("data/output/predictions_cmaq_hanoi_full.csv"),
        "plot": Path("data/output/comparison_cmaq_hanoi.png"),
        "map_data": Path("data/output/map_data_cmaq_hanoi.csv"),
        "raw_cols": ["pm25_cmaq_approx"],
        "default": False,
    },
}

# Bộ chính dùng gbtree
PRED_6POLL_PATH = Path("data/output/predictions_6pollutants_7days_gbtree.csv")
AQI_PATH = Path("data/output/vn_aqi_from_predictions_7days_gbtree.csv")
METRICS_6MODEL_PATH = Path("data/output/pollutant_metrics_6models_gbtree.csv")

# AQI tái tính từ quan trắc
RECOMP_AQI_PATH = Path("data/output/vn_aqi_recomputed_from_observations.csv")

META_PATH = Path("data/raw/stations/station_metadata.csv")
FINAL_FEATURE_TABLE_PATH = Path("data/processed/final_training_table_with_features_7days.csv")

POLLUTANT_META = {
    "pm25": {"label": "PM2.5", "obs": "pm25_obs", "pred": "pm25_pred"},
    "pm10": {"label": "PM10", "obs": "pm10_obs", "pred": "pm10_pred"},
    "o3": {"label": "O3", "obs": "o3_obs", "pred": "o3_pred"},
    "so2": {"label": "SO2", "obs": "so2_obs", "pred": "so2_pred"},
    "co": {"label": "CO", "obs": "co_obs", "pred": "co_pred"},
    "no2": {"label": "NO2", "obs": "no2_obs", "pred": "no2_pred"},
}

AQI_RANGES = {
    "Tốt": "0 - 50",
    "Trung bình": "51 - 100",
    "Kém": "101 - 150",
    "Xấu": "151 - 200",
    "Rất xấu": "201 - 300",
    "Nguy hại": "301 - 500",
}


# =========================
# HELPERS
# =========================
def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_nrmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    value_range = y_true.max() - y_true.min()
    if value_range == 0:
        return np.nan
    return float(rmse / value_range)


def fmt_metric(x, digits=3):
    if pd.isna(x):
        return "NaN"
    return f"{x:.{digits}f}"


def get_pm25_color(pm25: float) -> str:
    if pd.isna(pm25):
        return "gray"
    if pm25 <= 12:
        return "green"
    elif pm25 <= 35:
        return "orange"
    elif pm25 <= 55:
        return "red"
    return "darkred"


def get_pm25_level_text(pm25: float) -> str:
    if pd.isna(pm25):
        return "Không có dữ liệu"
    if pm25 <= 12:
        return "Thấp"
    elif pm25 <= 35:
        return "Trung bình"
    elif pm25 <= 55:
        return "Cao"
    return "Rất cao"


def get_aqi_color(level: str) -> str:
    mapping = {
        "Tốt": "#00E400",
        "Trung bình": "#FFFF00",
        "Kém": "#FF7E00",
        "Xấu": "#FF0000",
        "Rất xấu": "#8F3F97",
        "Nguy hại": "#7E0023",
    }
    return mapping.get(level, "#D1D5DB")


def get_aqi_badge_text_color(level: str) -> str:
    dark_levels = {"Xấu", "Rất xấu", "Nguy hại"}
    return "#ffffff" if level in dark_levels else "#111827"


def get_aqi_level_from_value(aqi):
    if pd.isna(aqi):
        return None
    aqi = float(aqi)
    if aqi <= 50:
        return "Tốt"
    elif aqi <= 100:
        return "Trung bình"
    elif aqi <= 150:
        return "Kém"
    elif aqi <= 200:
        return "Xấu"
    elif aqi <= 300:
        return "Rất xấu"
    return "Nguy hại"


def render_aqi_status_card(title: str, aqi_value, level_text: str, dominant_text: str = "", extra_text: str = ""):
    if pd.isna(aqi_value) or level_text is None:
        st.info(f"{title}: chưa có đủ dữ liệu để xác định trạng thái AQI.")
        return

    level_color = get_aqi_color(level_text)
    badge_text_color = get_aqi_badge_text_color(level_text)
    aqi_range_text = AQI_RANGES.get(level_text, "N/A")
    dominant_part = f" | Chất chi phối: <b>{dominant_text}</b>" if dominant_text else ""
    extra_part = f"<div style='font-size:14px;color:#6b7280;margin-top:8px;'>{extra_text}</div>" if extra_text else ""

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-left:10px solid {level_color};
            border-radius:16px;
            padding:18px 22px;
            margin-bottom:18px;
            box-shadow:0 2px 8px rgba(0,0,0,0.05);
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:14px;color:#6b7280;margin-bottom:6px;">{title}</div>
                    <div style="font-size:30px;font-weight:800;color:#111827;line-height:1.2;">
                        VN_AQI {int(aqi_value)}
                    </div>
                    <div style="font-size:16px;color:#374151;margin-top:8px;">
                        Mức: <b>{level_text}</b> | Khoảng AQI: <b>{aqi_range_text}</b>{dominant_part}
                    </div>
                    {extra_part}
                </div>
                <div style="
                    background:{level_color};
                    color:{badge_text_color};
                    font-weight:800;
                    font-size:18px;
                    padding:12px 22px;
                    border-radius:999px;
                    min-width:150px;
                    text-align:center;
                    box-shadow:inset 0 -1px 0 rgba(0,0,0,0.08);
                ">
                    {level_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def add_legend_pm25(map_obj: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 240px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="font-weight:600; margin-bottom:8px;">Chú giải PM2.5</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:green;border-radius:50%;margin-right:8px;"></span>≤ 12: Thấp</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:orange;border-radius:50%;margin-right:8px;"></span>12 - 35: Trung bình</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:red;border-radius:50%;margin-right:8px;"></span>35 - 55: Cao</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:darkred;border-radius:50%;margin-right:8px;"></span>> 55: Rất cao</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def add_legend_aqi(map_obj: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 260px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="font-weight:600; margin-bottom:8px;">Chú giải VN_AQI</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#00E400;border-radius:50%;margin-right:8px;"></span>0 - 50: Tốt</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FFFF00;border-radius:50%;margin-right:8px;"></span>51 - 100: Trung bình</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FF7E00;border-radius:50%;margin-right:8px;"></span>101 - 150: Kém</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FF0000;border-radius:50%;margin-right:8px;"></span>151 - 200: Xấu</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#8F3F97;border-radius:50%;margin-right:8px;"></span>201 - 300: Rất xấu</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#7E0023;border-radius:50%;margin-right:8px;"></span>301 - 500: Nguy hại</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def render_2day_station_map(day_df, title_text, map_key):
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:700;color:#0b74c9;margin-bottom:10px;'>{title_text}</div>",
        unsafe_allow_html=True,
    )

    if day_df.empty:
        st.info("Không có dữ liệu cho ngày này.")
        return

    valid_points = day_df.dropna(subset=["lat", "lon"]).copy()
    if valid_points.empty:
        st.warning("Không có tọa độ hợp lệ để vẽ bản đồ.")
        return

    center_lat = valid_points["lat"].mean()
    center_lon = valid_points["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )
    Fullscreen().add_to(m)
    add_legend_aqi(m)

    for _, row in valid_points.iterrows():
        aqi_val = row.get("vn_aqi_hour", np.nan)
        level = get_aqi_level_from_value(aqi_val)
        color = get_aqi_color(level)

        station_name = row.get("station_name", row.get("station_id", "N/A"))
        station_id = row.get("station_id", "N/A")
        aqi_text = str(int(aqi_val)) if pd.notna(aqi_val) else "NaN"
        level_text = level if level else "Không có dữ liệu"

        popup_html = f"""
        <div style='font-size:14px; line-height:1.5; min-width:220px;'>
            <b>Trạm:</b> {station_name}<br>
            <b>Mã trạm:</b> {station_id}<br>
            <b>AQI ngày:</b> {aqi_text}<br>
            <b>Mức:</b> {level_text}
        </div>
        """

        tooltip_text = f"{station_name} | AQI {aqi_text} | {level_text}"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=18,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=3,
        ).add_to(m)

        folium.map.Marker(
            [row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:13px;
                    font-weight:700;
                    color:#111827;
                    background:rgba(255,255,255,0.92);
                    border:1px solid #d1d5db;
                    border-radius:8px;
                    padding:2px 8px;
                    white-space:nowrap;
                    transform: translate(18px, -10px);
                ">
                    {station_name}<br>AQI: {aqi_text}
                </div>
                """
            )
        ).add_to(m)

    st_folium(m, width=650, height=520, key=map_key)


def ensure_map_data(pred_full_path: Path, map_data_path: Path, model_name: str):
    if map_data_path.exists() or not pred_full_path.exists():
        return
    if not META_PATH.exists():
        return

    pred_df = pd.read_csv(pred_full_path, parse_dates=["timestamp"])
    meta_df = pd.read_csv(META_PATH)

    join_cols = ["station_id"]
    if "station_name" in pred_df.columns and "station_name" in meta_df.columns:
        join_cols = ["station_id", "station_name"]

    merged = pred_df.merge(meta_df, on=join_cols, how="left")

    if model_name == "HYBRID":
        keep_cols = [
            "timestamp", "station_id", "station_name", "lat", "lon",
            "pm25_obs", "silam_pm25", "cmaq_pm25_approx", "pm25_pred_corrected"
        ]
    elif model_name == "SILAM":
        keep_cols = [
            "timestamp", "station_id", "station_name", "lat", "lon",
            "pm25_obs", "silam_pm25", "pm25_pred_corrected"
        ]
    else:
        keep_cols = [
            "timestamp", "station_id", "station_name", "lat", "lon",
            "pm25_obs", "pm25_cmaq_approx", "pm25_pred_corrected"
        ]

    keep_cols = [c for c in keep_cols if c in merged.columns]
    map_data_path.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_csv(map_data_path, index=False)


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_csv_parse_time(path: Path, time_col: str = "timestamp"):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


@st.cache_data
def load_meta():
    if not META_PATH.exists():
        return None
    return pd.read_csv(META_PATH)


@st.cache_data
def load_actual_aqi_table():
    if not FINAL_FEATURE_TABLE_PATH.exists():
        return None
    df = pd.read_csv(FINAL_FEATURE_TABLE_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    keep_cols = ["timestamp", "station_id", "vn_aqi"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def metric_card(title: str, rows: list[str], note: str = ""):
    rows_html = "".join([f'<div class="metric-row">{r}</div>' for r in rows])
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            {rows_html}
            {note_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_merge_station_name(df: pd.DataFrame, meta_df: pd.DataFrame):
    if df is None or meta_df is None:
        return df
    if "station_name" in df.columns:
        return df
    keep_meta = [c for c in ["station_id", "station_name", "lat", "lon"] if c in meta_df.columns]
    return df.merge(meta_df[keep_meta], on="station_id", how="left")


def calc_compare_metrics(df: pd.DataFrame, actual_col: str, pred_col: str):
    if actual_col not in df.columns or pred_col not in df.columns:
        return None
    sub = df.dropna(subset=[actual_col, pred_col]).copy()
    if sub.empty:
        return None
    mae = mean_absolute_error(sub[actual_col], sub[pred_col])
    rmse = calc_rmse(sub[actual_col], sub[pred_col])
    nrmse = calc_nrmse(sub[actual_col], sub[pred_col])
    return {"n": len(sub), "mae": mae, "rmse": rmse, "nrmse": nrmse}


def render_station_aqi_side_panel(sub_df: pd.DataFrame, mode: str, value_col: str, level_col: str):
    st.markdown("### Mức độ theo 3 trạm")
    if sub_df.empty:
        st.info("Không có dữ liệu bản đồ ở thời điểm đã chọn.")
        return

    sorted_sub = sub_df.sort_values("station_id")
    for _, row in sorted_sub.iterrows():
        aqi_value = row.get(value_col, np.nan)
        level_value = row.get(level_col, None)
        color = get_aqi_color(level_value)

        station_name = row.get("station_name", row.get("station_id", "N/A"))
        station_id = row.get("station_id", "N/A")
        aqi_text = str(int(aqi_value)) if pd.notna(aqi_value) else "NaN"
        level_text = level_value if level_value is not None else "Không có dữ liệu"

        st.markdown(
            f"""
            <div class="legend-panel">
                <div style="font-weight:700; margin-bottom:6px;">{station_name}</div>
                <div style="color:#6b7280; font-size:13px; margin-bottom:8px;">{station_id}</div>
                <div class="legend-item">
                    <div>
                        <span class="legend-dot" style="background:{color}; width:14px; height:14px; border-radius:50%; display:inline-block; margin-right:8px;"></span>
                        <span><b>{level_text}</b></span>
                    </div>
                    <div><b>{aqi_text}</b></div>
                </div>
                <div style="font-size:12px; color:#6b7280; margin-top:6px;">{mode}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
def get_aqi_bg_hex(aqi_value):
    level = get_aqi_level_from_value(aqi_value)
    return get_aqi_color(level)


def get_station_display_name(station_id):
    mapping = {
        "HN_NVC": "556 Nguyễn Văn Cừ",
        "HN_NC": "Công viên Nhân Chính",
        "HN_GP": "Số 1 Giải Phóng",
    }
    return mapping.get(station_id, station_id)
def get_aqi_level_emoji(level):
    mapping = {
        "Tốt": "🟢",
        "Trung bình": "🟡",
        "Kém": "🟠",
        "Xấu": "🔴",
        "Rất xấu": "🟣",
        "Nguy hại": "🟤",
    }
    return mapping.get(level, "⚪")


def render_2day_station_cards(day_df, title_text):
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:700;color:#0b74c9;margin-bottom:10px;'>{title_text}</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        if day_df.empty:
            st.info("Không có dữ liệu cho ngày này.")
            return

        ordered_df = day_df.sort_values("vn_aqi_hour", ascending=False).copy()

        for _, row in ordered_df.iterrows():
            station_name = row.get("station_name", get_station_display_name(row.get("station_id")))
            station_id = row.get("station_id", "N/A")
            aqi_val = row.get("vn_aqi_hour", np.nan)
            level = get_aqi_level_from_value(aqi_val)
            emoji = get_aqi_level_emoji(level)
            aqi_text = str(int(aqi_val)) if pd.notna(aqi_val) else "NaN"

            card = st.container(border=True)
            left, right = card.columns([2.6, 1])

            with left:
                st.markdown(f"**{station_name}**")
                st.caption(station_id)
                st.write(f"{emoji} **{level if level else 'Không có dữ liệu'}**")

            with right:
                st.metric("AQI", aqi_text)


def render_2day_ranking_table(day_df, date_label):
    st.markdown(
        f"<div style='text-align:center;font-size:20px;font-weight:800;margin-top:14px;margin-bottom:10px;'>Xếp hạng chất lượng không khí<br>{date_label}</div>",
        unsafe_allow_html=True,
    )

    if day_df.empty:
        st.info("Không có dữ liệu để xếp hạng.")
        return

    rank_df = day_df.copy()
    rank_df["Trạm"] = rank_df["station_id"].apply(get_station_display_name)
    rank_df["AQI"] = rank_df["vn_aqi_hour"].round().astype("Int64")
    rank_df["Mức"] = rank_df["AQI"].apply(get_aqi_level_from_value)
    rank_df = rank_df.sort_values("AQI", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "Xếp hạng", rank_df.index + 1)

    display_df = rank_df[["Xếp hạng", "Trạm", "AQI", "Mức"]].copy()

    def style_aqi(val):
        if pd.isna(val):
            return ""
        level = get_aqi_level_from_value(val)
        bg = get_aqi_color(level)
        fg = "#ffffff" if level in {"Xấu", "Rất xấu", "Nguy hại"} else "#111827"
        return f"background-color: {bg}; color: {fg}; font-weight: 700; text-align: center;"

    styler = (
        display_df.style
        .map(style_aqi, subset=["AQI"])
        .set_properties(**{"text-align": "center"}, subset=["Xếp hạng", "AQI", "Mức"])
    )

    st.dataframe(styler, use_container_width=True, hide_index=True)


def prepare_2day_forecast_table(aqi_df):
    """
    Lấy đúng 2 ngày 30/12/2025 và 31/12/2025.
    Giá trị đại diện mỗi trạm trong ngày là AQI lớn nhất của ngày đó.
    Giữ luôn lat/lon đã được merge sẵn từ metadata.
    """
    if aqi_df is None or aqi_df.empty:
        return None, []

    df = aqi_df.copy()
    df = df.dropna(subset=["timestamp"])
    df["date_only"] = pd.to_datetime(df["timestamp"]).dt.date

    valid = df.dropna(subset=["vn_aqi_hour"]).copy()
    if valid.empty:
        return None, []

    target_dates = [
        pd.to_datetime("2025-12-30").date(),
        pd.to_datetime("2025-12-31").date(),
    ]

    valid = valid[valid["date_only"].isin(target_dates)].copy()
    if valid.empty:
        return None, []

    # Giữ station_name, lat, lon đã có sẵn từ aqi_df
    keep_cols = ["date_only", "station_id", "station_name", "lat", "lon", "vn_aqi_hour"]
    keep_cols = [c for c in keep_cols if c in valid.columns]
    valid = valid[keep_cols].copy()

    # Lấy AQI lớn nhất trong ngày cho mỗi trạm
    idx = valid.groupby(["date_only", "station_id"])["vn_aqi_hour"].idxmax()
    daily = valid.loc[idx].copy().reset_index(drop=True)

    return daily, target_dates

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main .block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    min-height: 170px;
    margin-bottom: 20px;
}
.metric-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #1f2937;
}
.metric-row {
    font-size: 16px;
    margin-bottom: 8px;
    color: #374151;
}
.metric-note {
    margin-top: 10px;
    font-size: 13px;
    color: #6b7280;
}
.big-title {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
}
.legend-panel {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}
.legend-item {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    padding:10px 0;
    border-bottom:1px solid #e5e7eb;
}
.legend-item:last-child {
    border-bottom:none;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
pred_6 = load_csv_parse_time(PRED_6POLL_PATH)
aqi_df = load_csv_parse_time(AQI_PATH)
metrics_6 = load_csv(METRICS_6MODEL_PATH)
meta_df = load_meta()
aqi_actual_df = load_actual_aqi_table()
recomp_df = load_csv_parse_time(RECOMP_AQI_PATH)

for name, cfg in MODEL_CONFIG.items():
    ensure_map_data(cfg["pred_full"], cfg["map_data"], name)

if aqi_df is not None:
    if meta_df is not None:
        aqi_df = safe_merge_station_name(aqi_df, meta_df)
    if aqi_actual_df is not None:
        aqi_df = aqi_df.merge(
            aqi_actual_df.rename(columns={"vn_aqi": "vn_aqi_actual"}),
            on=["timestamp", "station_id"],
            how="left"
        )
        if "vn_aqi_actual" in aqi_df.columns and "level_actual" not in aqi_df.columns:
            aqi_df["level_actual"] = aqi_df["vn_aqi_actual"].apply(get_aqi_level_from_value)

if recomp_df is not None:
    if meta_df is not None:
        recomp_df = safe_merge_station_name(recomp_df, meta_df)
    if "vn_aqi_recomputed_from_obs" in recomp_df.columns and "level_recomputed" not in recomp_df.columns:
        recomp_df["level_recomputed"] = recomp_df["vn_aqi_recomputed_from_obs"].apply(get_aqi_level_from_value)
    if aqi_actual_df is not None and "vn_aqi" not in recomp_df.columns:
        recomp_df = recomp_df.merge(aqi_actual_df, on=["timestamp", "station_id"], how="left")
    if aqi_df is not None and "vn_aqi_hour" in aqi_df.columns:
        recomp_df = recomp_df.merge(
            aqi_df[["timestamp", "station_id", "vn_aqi_hour"]].rename(columns={"vn_aqi_hour": "vn_aqi_pred_from_model"}),
            on=["timestamp", "station_id"],
            how="left"
        )

if pred_6 is not None and meta_df is not None:
    pred_6 = safe_merge_station_name(pred_6, meta_df)

# =========================
# HEADER
# =========================
st.markdown('<div class="big-title">Dashboard dự báo chất lượng không khí Hà Nội</div>', unsafe_allow_html=True)
st.caption("Phiên bản chính đang dùng mô hình XGBoost gbtree cho dự báo 6 chất và VN_AQI.")

status_cols = st.columns(5)
status_cols[0].success("Dữ liệu trạm: OK")
status_cols[1].success("SILAM: OK")
status_cols[2].success("CMAQ: OK")
status_cols[3].success("AI gbtree: OK")
status_cols[4].success("VN_AQI gbtree: OK")

main_tabs = st.tabs([
    "Dự báo 48h",
    "VN_AQI",
    "VN_AQI tái tính",
    "Tổng quan",
    "Dự báo 6 chất",
    "PM2.5",
    "Bảng dữ liệu"
])
# =========================
# TAB 0 - DỰ BÁO 48H
# =========================
with main_tabs[0]:
    st.subheader("Dự báo chất lượng không khí 2 ngày cho 3 trạm")
    st.caption("Hiển thị nhanh mức AQI dự báo theo ngày cho 3 trạm Hà Nội, dùng AQI lớn nhất trong ngày để đại diện mức cảnh báo.")

    daily_forecast_df, selected_dates = prepare_2day_forecast_table(aqi_df)

    if daily_forecast_df is None or len(selected_dates) == 0:
        st.warning("Chưa có đủ dữ liệu AQI dự báo để tạo tab Dự báo 48h.")
    else:
        day1 = selected_dates[0]
        day2 = selected_dates[1] if len(selected_dates) > 1 else None

        day1_df = daily_forecast_df[daily_forecast_df["date_only"] == day1].copy()
        day2_df = daily_forecast_df[daily_forecast_df["date_only"] == day2].copy() if day2 else pd.DataFrame()

        col1, col2 = st.columns(2)

        with col1:
            render_2day_station_map(
                day1_df,
                f"Dự báo chất lượng không khí ngày: {pd.to_datetime(str(day1)).strftime('%d/%m/%Y')}",
                map_key=f"forecast_map_{day1}"
            )
            render_2day_ranking_table(
                day1_df,
                f"Ngày {pd.to_datetime(str(day1)).strftime('%d/%m/%Y')}"
            )

        with col2:
            if day2 is not None:
                render_2day_station_map(
                    day2_df,
                    f"Dự báo chất lượng không khí ngày: {pd.to_datetime(str(day2)).strftime('%d/%m/%Y')}",
                    map_key=f"forecast_map_{day2}"
                )
                render_2day_ranking_table(
                    day2_df,
                    f"Ngày {pd.to_datetime(str(day2)).strftime('%d/%m/%Y')}"
                )
            else:
                st.info("Hiện chỉ có 1 ngày đủ dữ liệu dự báo.")
# =========================
# TAB 1 - VN_AQI DỰ BÁO
# =========================
with main_tabs[1]:
    st.subheader("VN_AQI từ dự báo 6 chất")

    if aqi_df is None or aqi_df.empty:
        st.warning("Chưa có file vn_aqi_from_predictions_7days_gbtree.csv")
    else:
        stations = sorted(aqi_df["station_id"].dropna().unique().tolist())
        station_sel = st.selectbox("Chọn trạm AQI", stations, key="aqi_station")

        station_df = aqi_df[aqi_df["station_id"] == station_sel].sort_values("timestamp").copy()
        valid_pred_df = station_df.dropna(subset=["vn_aqi_hour"]).copy()
        compare_df = station_df.dropna(subset=["vn_aqi_hour", "vn_aqi_actual"]).copy() if "vn_aqi_actual" in station_df.columns else pd.DataFrame()

        top1, top2, top3, top4 = st.columns(4)
        if not valid_pred_df.empty:
            latest = valid_pred_df.iloc[-1]
            top1.metric("VN_AQI dự báo mới nhất", int(latest["vn_aqi_hour"]))
            top2.metric("Chất chi phối", str(latest["dominant_pollutant"]))
            top3.metric("Số giờ có AQI", int(station_df["vn_aqi_hour"].notna().sum()))
            top4.metric("Số giờ thiếu AQI", int(station_df["vn_aqi_hour"].isna().sum()))

            render_aqi_status_card(
                title="Trạng thái chất lượng không khí hiện tại",
                aqi_value=latest["vn_aqi_hour"],
                level_text=latest["level"],
                dominant_text=latest["dominant_pollutant"],
                extra_text="AQI được tính từ 6 chất sau dự báo AI gbtree."
            )

        st.subheader("Đánh giá VN_AQI thật vs dự báo")
        if not compare_df.empty:
            mae_aqi = mean_absolute_error(compare_df["vn_aqi_actual"], compare_df["vn_aqi_hour"])
            rmse_aqi = calc_rmse(compare_df["vn_aqi_actual"], compare_df["vn_aqi_hour"])
            nrmse_aqi = calc_nrmse(compare_df["vn_aqi_actual"], compare_df["vn_aqi_hour"])

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(
                    "VN_AQI thực tế",
                    [
                        f"<b>Số giờ có dữ liệu:</b> {int(station_df['vn_aqi_actual'].notna().sum()) if 'vn_aqi_actual' in station_df.columns else 0}",
                        f"<b>Giá trị TB:</b> {fmt_metric(station_df['vn_aqi_actual'].dropna().mean(), 1) if station_df['vn_aqi_actual'].notna().sum() > 0 else 'NaN'}",
                        f"<b>Mới nhất:</b> {int(station_df['vn_aqi_actual'].dropna().iloc[-1]) if station_df['vn_aqi_actual'].notna().sum() > 0 else 'NaN'}",
                    ],
                    "Lấy từ dữ liệu nguồn."
                )
            with c2:
                metric_card(
                    "VN_AQI từ dự báo 6 chất",
                    [
                        f"<b>Số giờ có dữ liệu:</b> {int(station_df['vn_aqi_hour'].notna().sum())}",
                        f"<b>Giá trị TB:</b> {fmt_metric(station_df['vn_aqi_hour'].dropna().mean(), 1) if station_df['vn_aqi_hour'].notna().sum() > 0 else 'NaN'}",
                        f"<b>Mới nhất:</b> {int(station_df['vn_aqi_hour'].dropna().iloc[-1]) if station_df['vn_aqi_hour'].notna().sum() > 0 else 'NaN'}",
                    ],
                    "Tính từ 6 chất sau dự báo AI gbtree."
                )
            with c3:
                metric_card(
                    "Sai số AQI",
                    [
                        f"<b>MAE:</b> {fmt_metric(mae_aqi)}",
                        f"<b>RMSE:</b> {fmt_metric(rmse_aqi)}",
                        f"<b>NRMSE:</b> {fmt_metric(nrmse_aqi)}",
                    ],
                    "So với AQI nguồn trong dữ liệu."
                )

        left, right = st.columns([1.45, 1])
        with left:
            st.subheader("Biểu đồ VN_AQI theo giờ")
            chart_cols = ["vn_aqi_hour"]
            if "vn_aqi_actual" in station_df.columns:
                chart_cols.append("vn_aqi_actual")
            chart_df = station_df[["timestamp"] + chart_cols].copy().set_index("timestamp")
            chart_df = chart_df.rename(columns={
                "vn_aqi_hour": "VN_AQI dự báo",
                "vn_aqi_actual": "VN_AQI thực tế"
            })
            st.line_chart(chart_df)

        with right:
            st.subheader("Khuyến nghị sức khỏe")
            if not valid_pred_df.empty:
                st.info(str(latest["health_text"]))
                st.warning(str(latest["recommendation"]))

        st.subheader("Bản đồ VN_AQI")
        time_options = sorted(aqi_df["timestamp"].astype(str).unique().tolist())
        c_map, c_side = st.columns([3.2, 1.25])

        with c_map:
            left_sel, right_sel = st.columns([2, 1])
            with left_sel:
                selected_time = st.selectbox("Chọn thời điểm AQI", time_options, key="aqi_time")
            with right_sel:
                aqi_map_mode = st.selectbox(
                    "Tô màu theo",
                    ["VN_AQI dự báo", "VN_AQI thực tế"],
                    key="aqi_color_mode"
                )

            sub = aqi_df[aqi_df["timestamp"].astype(str) == selected_time].copy()
            m = folium.Map(location=[21.02, 105.84], zoom_start=11, tiles="OpenStreetMap")
            Fullscreen().add_to(m)
            add_legend_aqi(m)

            for _, row in sub.iterrows():
                if pd.isna(row.get("lat", np.nan)) or pd.isna(row.get("lon", np.nan)):
                    continue

                if aqi_map_mode == "VN_AQI dự báo":
                    aqi_value = row.get("vn_aqi_hour", np.nan)
                    level_value = row.get("level", None)
                else:
                    aqi_value = row.get("vn_aqi_actual", np.nan)
                    level_value = row.get("level_actual", None)

                color = get_aqi_color(level_value)
                popup_html = f"""
                <div style='font-size:14px; line-height:1.5;'>
                    <b>Trạm:</b> {row.get('station_name', row.get('station_id', 'N/A'))}<br>
                    <b>Mã trạm:</b> {row.get('station_id', 'N/A')}<br>
                    <b>VN_AQI thực tế:</b> {int(row['vn_aqi_actual']) if pd.notna(row.get('vn_aqi_actual', np.nan)) else 'NaN'}<br>
                    <b>VN_AQI dự báo:</b> {int(row['vn_aqi_hour']) if pd.notna(row.get('vn_aqi_hour', np.nan)) else 'NaN'}<br>
                    <b>Mức đang tô màu:</b> {level_value if level_value is not None else 'NaN'}
                </div>
                """
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=12,
                    popup=folium.Popup(popup_html, max_width=340),
                    tooltip=f"{row.get('station_name', row.get('station_id'))} | {aqi_value if pd.notna(aqi_value) else 'NaN'}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    weight=2
                ).add_to(m)

            st_folium(m, width=980, height=560, key=f"map_aqi_pred_{selected_time}_{aqi_map_mode}")

        with c_side:
            if aqi_map_mode == "VN_AQI dự báo":
                render_station_aqi_side_panel(sub, aqi_map_mode, "vn_aqi_hour", "level")
            else:
                render_station_aqi_side_panel(sub, aqi_map_mode, "vn_aqi_actual", "level_actual")

        st.subheader("Bảng AQI theo giờ")
        show_cols = [
            "timestamp", "station_id", "station_name",
            "vn_aqi_actual", "vn_aqi_hour",
            "dominant_pollutant", "level", "level_actual",
            "aqi_pm25", "aqi_pm10", "aqi_o3", "aqi_so2", "aqi_co", "aqi_no2",
            "pm25_nowcast", "pm10_nowcast"
        ]
        show_cols = [c for c in show_cols if c in station_df.columns]
        st.dataframe(station_df[show_cols], use_container_width=True)

# =========================
# TAB 2 - VN_AQI TÁI TÍNH
# =========================
with main_tabs[2]:
    st.subheader("VN_AQI tái tính từ quan trắc")

    if recomp_df is None or recomp_df.empty:
        st.warning("Chưa có file vn_aqi_recomputed_from_observations.csv")
    else:
        stations = sorted(recomp_df["station_id"].dropna().unique().tolist())
        station_sel = st.selectbox("Chọn trạm AQI tái tính", stations, key="recomp_station")

        station_df = recomp_df[recomp_df["station_id"] == station_sel].sort_values("timestamp").copy()
        valid_recomp_df = station_df.dropna(subset=["vn_aqi_recomputed_from_obs"]).copy()

        compare_source_df = station_df.dropna(subset=["vn_aqi_recomputed_from_obs", "vn_aqi"]).copy() if "vn_aqi" in station_df.columns else pd.DataFrame()
        compare_pred_df = station_df.dropna(subset=["vn_aqi_recomputed_from_obs", "vn_aqi_pred_from_model"]).copy() if "vn_aqi_pred_from_model" in station_df.columns else pd.DataFrame()

        top1, top2, top3, top4 = st.columns(4)
        if not valid_recomp_df.empty:
            latest = valid_recomp_df.iloc[-1]
            top1.metric("VN_AQI tái tính mới nhất", int(latest["vn_aqi_recomputed_from_obs"]))
            top2.metric("Chất chi phối tái tính", str(latest["dominant_pollutant_recomputed"]))
            top3.metric("Số giờ có AQI tái tính", int(station_df["vn_aqi_recomputed_from_obs"].notna().sum()))
            top4.metric("Số giờ thiếu AQI tái tính", int(station_df["vn_aqi_recomputed_from_obs"].isna().sum()))

            level_latest = latest.get("level_recomputed", None)
            render_aqi_status_card(
                title="Trạng thái AQI tái tính từ quan trắc",
                aqi_value=latest["vn_aqi_recomputed_from_obs"],
                level_text=level_latest,
                dominant_text=latest.get("dominant_pollutant_recomputed", ""),
                extra_text="AQI được tính lại trực tiếp từ các cột quan trắc hiện có."
            )

        st.subheader("Đánh giá AQI tái tính")

        c1, c2, c3 = st.columns(3)

        with c1:
            if not compare_source_df.empty:
                mae = mean_absolute_error(compare_source_df["vn_aqi"], compare_source_df["vn_aqi_recomputed_from_obs"])
                rmse = calc_rmse(compare_source_df["vn_aqi"], compare_source_df["vn_aqi_recomputed_from_obs"])
                nrmse = calc_nrmse(compare_source_df["vn_aqi"], compare_source_df["vn_aqi_recomputed_from_obs"])
                metric_card(
                    "Tái tính vs AQI nguồn",
                    [
                        f"<b>Số giờ chồng lắp:</b> {len(compare_source_df)}",
                        f"<b>MAE:</b> {fmt_metric(mae)}",
                        f"<b>RMSE:</b> {fmt_metric(rmse)}",
                        f"<b>NRMSE:</b> {fmt_metric(nrmse)}",
                    ],
                    "Đánh giá mức lệch giữa AQI tái tính và AQI nguồn."
                )
            else:
                metric_card("Tái tính vs AQI nguồn", ["<b>Số giờ chồng lắp:</b> 0", "<b>MAE:</b> NaN", "<b>RMSE:</b> NaN", "<b>NRMSE:</b> NaN"])

        with c2:
            if not compare_pred_df.empty:
                mae = mean_absolute_error(compare_pred_df["vn_aqi_recomputed_from_obs"], compare_pred_df["vn_aqi_pred_from_model"])
                rmse = calc_rmse(compare_pred_df["vn_aqi_recomputed_from_obs"], compare_pred_df["vn_aqi_pred_from_model"])
                nrmse = calc_nrmse(compare_pred_df["vn_aqi_recomputed_from_obs"], compare_pred_df["vn_aqi_pred_from_model"])
                metric_card(
                    "Tái tính vs AQI dự báo",
                    [
                        f"<b>Số giờ chồng lắp:</b> {len(compare_pred_df)}",
                        f"<b>MAE:</b> {fmt_metric(mae)}",
                        f"<b>RMSE:</b> {fmt_metric(rmse)}",
                        f"<b>NRMSE:</b> {fmt_metric(nrmse)}",
                    ],
                    "Đánh giá AQI dự báo so với AQI tái tính từ quan trắc."
                )
            else:
                metric_card("Tái tính vs AQI dự báo", ["<b>Số giờ chồng lắp:</b> 0", "<b>MAE:</b> NaN", "<b>RMSE:</b> NaN", "<b>NRMSE:</b> NaN"])

        with c3:
            metric_card(
                "VN_AQI tái tính từ quan trắc",
                [
                    f"<b>Số giờ có dữ liệu:</b> {int(station_df['vn_aqi_recomputed_from_obs'].notna().sum())}",
                    f"<b>Giá trị TB:</b> {fmt_metric(station_df['vn_aqi_recomputed_from_obs'].dropna().mean(), 1) if station_df['vn_aqi_recomputed_from_obs'].notna().sum() > 0 else 'NaN'}",
                    f"<b>Mới nhất:</b> {int(station_df['vn_aqi_recomputed_from_obs'].dropna().iloc[-1]) if station_df['vn_aqi_recomputed_from_obs'].notna().sum() > 0 else 'NaN'}",
                ],
                "AQI tính lại trực tiếp từ các cột quan trắc."
            )

        left, right = st.columns([1.45, 1])
        with left:
            st.subheader("Biểu đồ AQI tái tính theo giờ")
            chart_cols = ["vn_aqi_recomputed_from_obs"]
            if "vn_aqi" in station_df.columns:
                chart_cols.append("vn_aqi")
            if "vn_aqi_pred_from_model" in station_df.columns:
                chart_cols.append("vn_aqi_pred_from_model")

            chart_df = station_df[["timestamp"] + chart_cols].copy().set_index("timestamp")
            chart_df = chart_df.rename(columns={
                "vn_aqi_recomputed_from_obs": "AQI tái tính",
                "vn_aqi": "AQI nguồn",
                "vn_aqi_pred_from_model": "AQI dự báo"
            })
            st.line_chart(chart_df)

        with right:
            st.subheader("Thông tin tái tính")
            st.info("Tab này dùng AQI tự tính lại từ chính dữ liệu quan trắc để kiểm tra xem AQI nguồn có cùng logic công thức hay không.")
            if not valid_recomp_df.empty:
                latest = valid_recomp_df.iloc[-1]
                st.write("Chất chi phối tái tính:", latest.get("dominant_pollutant_recomputed", "N/A"))
                st.write("Mức tái tính:", latest.get("level_recomputed", "N/A"))

        st.subheader("Bản đồ AQI tái tính")
        time_options = sorted(recomp_df["timestamp"].astype(str).unique().tolist())
        c_map, c_side = st.columns([3.2, 1.25])

        with c_map:
            left_sel, right_sel = st.columns([2, 1])
            with left_sel:
                selected_time = st.selectbox("Chọn thời điểm AQI tái tính", time_options, key="recomp_time")
            with right_sel:
                recomp_map_mode = st.selectbox(
                    "Tô màu theo",
                    ["AQI tái tính", "AQI nguồn", "AQI dự báo"],
                    key="recomp_color_mode"
                )

            sub = recomp_df[recomp_df["timestamp"].astype(str) == selected_time].copy()
            m = folium.Map(location=[21.02, 105.84], zoom_start=11, tiles="OpenStreetMap")
            Fullscreen().add_to(m)
            add_legend_aqi(m)

            for _, row in sub.iterrows():
                if pd.isna(row.get("lat", np.nan)) or pd.isna(row.get("lon", np.nan)):
                    continue

                if recomp_map_mode == "AQI tái tính":
                    aqi_value = row.get("vn_aqi_recomputed_from_obs", np.nan)
                    level_value = row.get("level_recomputed", None)
                elif recomp_map_mode == "AQI nguồn":
                    aqi_value = row.get("vn_aqi", np.nan)
                    level_value = get_aqi_level_from_value(aqi_value)
                else:
                    aqi_value = row.get("vn_aqi_pred_from_model", np.nan)
                    level_value = get_aqi_level_from_value(aqi_value)

                color = get_aqi_color(level_value)
                popup_html = f"""
                <div style='font-size:14px; line-height:1.5;'>
                    <b>Trạm:</b> {row.get('station_name', row.get('station_id', 'N/A'))}<br>
                    <b>AQI nguồn:</b> {int(row['vn_aqi']) if pd.notna(row.get('vn_aqi', np.nan)) else 'NaN'}<br>
                    <b>AQI tái tính:</b> {int(row['vn_aqi_recomputed_from_obs']) if pd.notna(row.get('vn_aqi_recomputed_from_obs', np.nan)) else 'NaN'}<br>
                    <b>AQI dự báo:</b> {int(row['vn_aqi_pred_from_model']) if pd.notna(row.get('vn_aqi_pred_from_model', np.nan)) else 'NaN'}<br>
                    <b>Mức đang tô màu:</b> {level_value if level_value is not None else 'NaN'}
                </div>
                """
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=12,
                    popup=folium.Popup(popup_html, max_width=340),
                    tooltip=f"{row.get('station_name', row.get('station_id'))} | {aqi_value if pd.notna(aqi_value) else 'NaN'}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    weight=2
                ).add_to(m)

            st_folium(m, width=980, height=560, key=f"map_aqi_recomp_{selected_time}_{recomp_map_mode}")

        with c_side:
            if recomp_map_mode == "AQI tái tính":
                sub["level_tmp"] = sub["vn_aqi_recomputed_from_obs"].apply(get_aqi_level_from_value)
                render_station_aqi_side_panel(sub, recomp_map_mode, "vn_aqi_recomputed_from_obs", "level_tmp")
            elif recomp_map_mode == "AQI nguồn":
                sub["level_tmp"] = sub["vn_aqi"].apply(get_aqi_level_from_value)
                render_station_aqi_side_panel(sub, recomp_map_mode, "vn_aqi", "level_tmp")
            else:
                sub["level_tmp"] = sub["vn_aqi_pred_from_model"].apply(get_aqi_level_from_value)
                render_station_aqi_side_panel(sub, recomp_map_mode, "vn_aqi_pred_from_model", "level_tmp")

        st.subheader("Bảng AQI tái tính theo giờ")
        show_cols = [
            "timestamp", "station_id", "station_name",
            "vn_aqi", "vn_aqi_recomputed_from_obs", "vn_aqi_pred_from_model",
            "diff_recomputed_minus_actual", "abs_diff", "diff_group",
            "dominant_pollutant_recomputed",
            "aqi_pm25_obs", "aqi_pm10_obs", "aqi_o3_obs", "aqi_so2_obs", "aqi_co_obs", "aqi_no2_obs",
            "pm25_nowcast_obs", "pm10_nowcast_obs"
        ]
        show_cols = [c for c in show_cols if c in station_df.columns]
        st.dataframe(station_df[show_cols], use_container_width=True)

# =========================
# TAB 3 - TỔNG QUAN
# =========================
with main_tabs[3]:
    st.subheader("Tổng quan hệ thống")

    c1, c2, c3, c4 = st.columns(4)
    n_station = meta_df["station_id"].nunique() if meta_df is not None else 0
    n_rows_pred = len(pred_6) if pred_6 is not None else 0
    n_rows_aqi = len(aqi_df) if aqi_df is not None else 0
    n_models = len(metrics_6) if metrics_6 is not None else 0

    c1.metric("Số trạm", n_station)
    c2.metric("Dòng dự báo 6 chất", n_rows_pred)
    c3.metric("Dòng VN_AQI dự báo", n_rows_aqi)
    c4.metric("Số model AI", n_models)

    if metrics_6 is not None and not metrics_6.empty:
        st.subheader("Đánh giá nhanh 6 model gbtree")
        show_cols = ["pollutant", "n_train", "n_test", "mae", "rmse", "nrmse", "r2"]
        show_cols = [c for c in show_cols if c in metrics_6.columns]
        st.dataframe(metrics_6[show_cols], use_container_width=True)

    if pred_6 is not None:
        st.subheader("Độ phủ dự báo theo 6 chất")
        pred_cols = [c for c in pred_6.columns if c.endswith("_pred")]
        cover = pd.DataFrame({
            "Chất": pred_cols,
            "Số dòng có dự báo": [int(pred_6[c].notna().sum()) for c in pred_cols]
        })
        st.bar_chart(cover.set_index("Chất"))

# =========================
# TAB 4 - DỰ BÁO 6 CHẤT
# =========================
with main_tabs[4]:
    st.subheader("Dự báo 6 chất ô nhiễm")

    if pred_6 is None or pred_6.empty:
        st.warning("Chưa có file predictions_6pollutants_7days_gbtree.csv")
    else:
        stations = sorted(pred_6["station_id"].dropna().unique().tolist())
        station_sel = st.selectbox("Chọn trạm", stations, key="poll_station")

        station_df = pred_6[pred_6["station_id"] == station_sel].sort_values("timestamp").copy()

        st.subheader("Độ phủ dự báo theo chất")
        pollutant_pred_cols = ["pm25_pred", "pm10_pred", "o3_pred", "so2_pred", "co_pred", "no2_pred"]
        cover_df = pd.DataFrame({
            "Chất": pollutant_pred_cols,
            "Số giờ có dự báo": [int(station_df[c].notna().sum()) if c in station_df.columns else 0 for c in pollutant_pred_cols]
        })
        st.bar_chart(cover_df.set_index("Chất"))

        st.subheader("Biểu đồ tổng hợp 6 chất dự báo")
        chart_cols = [c for c in pollutant_pred_cols if c in station_df.columns]
        chart_df = station_df[["timestamp"] + chart_cols].copy().set_index("timestamp")
        st.line_chart(chart_df)

        st.subheader("So sánh chi tiết từng chất")
        pollutant_key = st.selectbox(
            "Chọn chất để xem chi tiết",
            list(POLLUTANT_META.keys()),
            format_func=lambda x: POLLUTANT_META[x]["label"],
            key="pollutant_focus"
        )

        pmeta = POLLUTANT_META[pollutant_key]
        actual_col = pmeta["obs"]
        pred_col = pmeta["pred"]
        label = pmeta["label"]

        c1, c2 = st.columns([1.35, 1])
        with c1:
            chart_detail_cols = ["timestamp"]
            if actual_col in station_df.columns:
                chart_detail_cols.append(actual_col)
            if pred_col in station_df.columns:
                chart_detail_cols.append(pred_col)

            detail_df = station_df[chart_detail_cols].copy().set_index("timestamp")
            detail_df = detail_df.rename(columns={
                actual_col: f"{label} thực tế",
                pred_col: f"{label} dự báo"
            })
            st.line_chart(detail_df)

        with c2:
            metric_info = calc_compare_metrics(station_df, actual_col, pred_col)
            n_actual = int(station_df[actual_col].notna().sum()) if actual_col in station_df.columns else 0
            n_pred = int(station_df[pred_col].notna().sum()) if pred_col in station_df.columns else 0

            if metric_info is not None:
                metric_card(
                    f"Đánh giá {label}",
                    [
                        f"<b>Số giờ có thực tế:</b> {n_actual}",
                        f"<b>Số giờ có dự báo:</b> {n_pred}",
                        f"<b>Số giờ chồng lắp:</b> {metric_info['n']}",
                        f"<b>MAE:</b> {fmt_metric(metric_info['mae'])}",
                        f"<b>RMSE:</b> {fmt_metric(metric_info['rmse'])}",
                        f"<b>NRMSE:</b> {fmt_metric(metric_info['nrmse'])}",
                    ],
                    f"So sánh {label} dự báo với dữ liệu quan trắc thật tại trạm đã chọn."
                )
            else:
                metric_card(
                    f"Đánh giá {label}",
                    [
                        f"<b>Số giờ có thực tế:</b> {n_actual}",
                        f"<b>Số giờ có dự báo:</b> {n_pred}",
                        "<b>Số giờ chồng lắp:</b> 0",
                        "<b>MAE:</b> NaN",
                        "<b>RMSE:</b> NaN",
                        "<b>NRMSE:</b> NaN",
                    ],
                    f"Chưa đủ dữ liệu để so sánh {label} giữa thực tế và dự báo."
                )

        st.subheader(f"Bảng chi tiết {label}")
        detail_cols = ["timestamp", "station_id", "station_name", actual_col, pred_col]
        if pollutant_key == "pm25":
            detail_cols += ["silam_pm25", "cmaq_rh", "cmaq_ta"]
        elif pollutant_key == "pm10":
            detail_cols += ["silam_pm10", "cmaq_rh", "cmaq_ta"]
        elif pollutant_key == "o3":
            detail_cols += ["silam_o3", "cmaq_o3"]
        elif pollutant_key == "so2":
            detail_cols += ["silam_so2", "cmaq_so2"]
        elif pollutant_key == "co":
            detail_cols += ["silam_co", "cmaq_co"]
        elif pollutant_key == "no2":
            detail_cols += ["silam_no2", "cmaq_no2", "cmaq_no", "cmaq_nox"]

        detail_cols = [c for c in detail_cols if c in station_df.columns]
        st.dataframe(station_df[detail_cols], use_container_width=True)

# =========================
# TAB 5 - PM2.5
# =========================
with main_tabs[5]:
    st.subheader("Nhánh dashboard PM2.5 theo app cũ")

    model_names = list(MODEL_CONFIG.keys())
    default_index = next((i for i, name in enumerate(model_names) if MODEL_CONFIG[name]["default"]), 0)

    selected_model = st.radio(
        "Chọn nhánh mô hình",
        model_names,
        index=default_index,
        horizontal=True,
        key="legacy_model_radio"
    )

    cfg = MODEL_CONFIG[selected_model]
    pred_test_path = cfg["pred_test"]
    plot_path = cfg["plot"]
    map_data_path = cfg["map_data"]
    raw_cols = cfg["raw_cols"]

    pred_test_df = load_csv(pred_test_path)
    map_df = load_csv_parse_time(map_data_path)

    st.subheader("Chỉ số đánh giá")
    if pred_test_df is not None:
        mae_corr = mean_absolute_error(pred_test_df["pm25_obs"], pred_test_df["pm25_pred_corrected"])
        rmse_corr = calc_rmse(pred_test_df["pm25_obs"], pred_test_df["pm25_pred_corrected"])
        nrmse_corr = calc_nrmse(pred_test_df["pm25_obs"], pred_test_df["pm25_pred_corrected"])

        if selected_model == "HYBRID":
            mae_silam = mean_absolute_error(pred_test_df["pm25_obs"], pred_test_df["silam_pm25"])
            rmse_silam = calc_rmse(pred_test_df["pm25_obs"], pred_test_df["silam_pm25"])
            nrmse_silam = calc_nrmse(pred_test_df["pm25_obs"], pred_test_df["silam_pm25"])

            mae_cmaq = mean_absolute_error(pred_test_df["pm25_obs"], pred_test_df["cmaq_pm25_approx"])
            rmse_cmaq = calc_rmse(pred_test_df["pm25_obs"], pred_test_df["cmaq_pm25_approx"])
            nrmse_cmaq = calc_nrmse(pred_test_df["pm25_obs"], pred_test_df["cmaq_pm25_approx"])

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("SILAM raw", [f"<b>MAE:</b> {mae_silam:.3f}", f"<b>RMSE:</b> {rmse_silam:.3f}", f"<b>NRMSE:</b> {nrmse_silam:.3f}"])
            with c2:
                metric_card("CMAQ raw", [f"<b>MAE:</b> {mae_cmaq:.3f}", f"<b>RMSE:</b> {rmse_cmaq:.3f}", f"<b>NRMSE:</b> {nrmse_cmaq:.3f}"])
            with c3:
                metric_card("AI Hybrid", [f"<b>MAE:</b> {mae_corr:.3f}", f"<b>RMSE:</b> {rmse_corr:.3f}", f"<b>NRMSE:</b> {nrmse_corr:.3f}"])
        else:
            raw_col = raw_cols[0]
            mae_raw = mean_absolute_error(pred_test_df["pm25_obs"], pred_test_df[raw_col])
            rmse_raw = calc_rmse(pred_test_df["pm25_obs"], pred_test_df[raw_col])
            nrmse_raw = calc_nrmse(pred_test_df["pm25_obs"], pred_test_df[raw_col])

            c1, c2 = st.columns(2)
            with c1:
                metric_card(f"{selected_model} raw", [f"<b>MAE:</b> {mae_raw:.3f}", f"<b>RMSE:</b> {rmse_raw:.3f}", f"<b>NRMSE:</b> {nrmse_raw:.3f}"])
            with c2:
                metric_card("Sau AI hiệu chỉnh", [f"<b>MAE:</b> {mae_corr:.3f}", f"<b>RMSE:</b> {rmse_corr:.3f}", f"<b>NRMSE:</b> {nrmse_corr:.3f}"])

    st.subheader("Bản đồ PM2.5")
    if map_df is not None and not map_df.empty:
        time_options = sorted(map_df["timestamp"].astype(str).unique().tolist())
        left, right = st.columns([2, 1])
        with left:
            selected_time = st.selectbox("Chọn thời điểm hiển thị", time_options, key="legacy_time")
        with right:
            if selected_model == "HYBRID":
                color_mode = st.selectbox("Tô màu theo", ["PM2.5 sau AI hybrid", "PM2.5 thực tế", "SILAM thô", "CMAQ thô"], key="legacy_color_hybrid")
            elif selected_model == "SILAM":
                color_mode = st.selectbox("Tô màu theo", ["PM2.5 sau AI hiệu chỉnh", "PM2.5 thực tế", "SILAM thô"], key="legacy_color_silam")
            else:
                color_mode = st.selectbox("Tô màu theo", ["PM2.5 sau AI hiệu chỉnh", "PM2.5 thực tế", "CMAQ thô"], key="legacy_color_cmaq")

        sub = map_df[map_df["timestamp"].astype(str) == selected_time].copy()
        m = folium.Map(location=[21.02, 105.84], zoom_start=11, tiles="OpenStreetMap")
        Fullscreen().add_to(m)
        add_legend_pm25(m)

        for _, row in sub.iterrows():
            if color_mode in ["PM2.5 sau AI hybrid", "PM2.5 sau AI hiệu chỉnh"]:
                color_value = row.get("pm25_pred_corrected", np.nan)
            elif color_mode == "PM2.5 thực tế":
                color_value = row.get("pm25_obs", np.nan)
            elif color_mode == "SILAM thô":
                color_value = row.get("silam_pm25", np.nan)
            else:
                color_value = row.get("cmaq_pm25_approx", np.nan)

            color = get_pm25_color(color_value)
            level_text = get_pm25_level_text(color_value)

            popup_html = f"""
            <div style='font-size:14px; line-height:1.5;'>
                <b>Trạm:</b> {row.get('station_name', row.get('station_id', 'N/A'))}<br>
                <b>Mã trạm:</b> {row.get('station_id', 'N/A')}<br>
                <b>Thời gian:</b> {row.get('timestamp', 'N/A')}<br>
                <b>Mức:</b> {level_text}
            </div>
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=11,
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{row.get('station_name', row.get('station_id'))} | {color_value:.2f}" if pd.notna(color_value) else row.get('station_name', row.get('station_id')),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2
            ).add_to(m)

        st_folium(m, width=1150, height=560, key=f"map_pm25_{selected_model}_{selected_time}_{color_mode}")

    st.subheader("Biểu đồ so sánh")
    if plot_path.exists():
        st.image(str(plot_path), caption=f"Biểu đồ so sánh - {selected_model}")

# =========================
# TAB 6 - BẢNG DỮ LIỆU
# =========================
with main_tabs[6]:
    st.subheader("Bảng dữ liệu tổng hợp")

    option = st.selectbox(
        "Chọn bảng muốn xem",
        [
            "predictions_6pollutants_7days_gbtree",
            "vn_aqi_from_predictions_7days_gbtree",
            "vn_aqi_recomputed_from_observations",
            "pollutant_metrics_6models_gbtree"
        ],
        key="data_table_option"
    )

    if option == "predictions_6pollutants_7days_gbtree":
        if pred_6 is not None:
            st.dataframe(pred_6.head(200), use_container_width=True)
        else:
            st.warning("Không tìm thấy file predictions_6pollutants_7days_gbtree.csv")
    elif option == "vn_aqi_from_predictions_7days_gbtree":
        if aqi_df is not None:
            st.dataframe(aqi_df.head(200), use_container_width=True)
        else:
            st.warning("Không tìm thấy file vn_aqi_from_predictions_7days_gbtree.csv")
    elif option == "vn_aqi_recomputed_from_observations":
        if recomp_df is not None:
            st.dataframe(recomp_df.head(200), use_container_width=True)
        else:
            st.warning("Không tìm thấy file vn_aqi_recomputed_from_observations.csv")
    else:
        if metrics_6 is not None:
            st.dataframe(metrics_6, use_container_width=True)
        else:
            st.warning("Không tìm thấy file pollutant_metrics_6models_gbtree.csv")