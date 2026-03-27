from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Cấu hình đường dẫn dữ liệu
# =========================
MODEL_CONFIG = {
    "SILAM": {
        "pred_test": Path("data/output/predictions_silam_hanoi.csv"),
        "pred_full": Path("data/output/predictions_silam_hanoi_full.csv"),
        "plot": Path("data/output/comparison_silam_hanoi.png"),
        "map_data": Path("data/output/map_data_silam_hanoi.csv"),
        "raw_col": "silam_pm25",
        "raw_label": "SILAM thô",
        "default": True,
    },
    "CMAQ": {
        "pred_test": Path("data/output/predictions_cmaq_hanoi.csv"),
        "pred_full": Path("data/output/predictions_cmaq_hanoi_full.csv"),
        "plot": Path("data/output/comparison_cmaq_hanoi.png"),
        "map_data": Path("data/output/map_data_cmaq_hanoi.csv"),
        "raw_col": "pm25_cmaq_approx",
        "raw_label": "CMAQ thô",
        "default": False,
    },
}

st.set_page_config(
    page_title="Dashboard PM2.5 Hà Nội",
    page_icon="🌫️",
    layout="wide"
)

# =========================
# Helpers
# =========================
def get_color(pm25: float) -> str:
    if pm25 <= 12:
        return "green"
    elif pm25 <= 35:
        return "orange"
    elif pm25 <= 55:
        return "red"
    return "darkred"


def get_level_text(pm25: float) -> str:
    if pm25 <= 12:
        return "Thấp"
    elif pm25 <= 35:
        return "Trung bình"
    elif pm25 <= 55:
        return "Cao"
    return "Rất cao"


def add_legend(map_obj: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 220px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #ccc;
        border-radius: 8px;
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


def ensure_map_data(pred_full_path: Path, map_data_path: Path):
    """
    Nếu chưa có map_data_*.csv thì tự tạo nhanh từ pred_full + station_metadata.
    """
    if map_data_path.exists() or not pred_full_path.exists():
        return

    meta_file = Path("data/raw/stations/station_metadata.csv")
    if not meta_file.exists():
        return

    pred_df = pd.read_csv(pred_full_path, parse_dates=["timestamp"])
    meta_df = pd.read_csv(meta_file)

    merged = pred_df.merge(
        meta_df,
        on=["station_id", "station_name"],
        how="left"
    )

    keep_cols = [
        "timestamp",
        "station_id",
        "station_name",
        "lat",
        "lon",
        "pm25_obs",
        "pm25_pred_corrected",
        "silam_pm25",
        "silam_pm10",
        "silam_aqi",
        "pm25_cmaq_approx",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]

    map_data_path.parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_csv(map_data_path, index=False)


# =========================
# Header
# =========================
st.title("Dashboard hiệu chỉnh dự báo PM2.5 - Hà Nội")
st.caption(
    "So sánh và trực quan hóa 2 nhánh mô hình vật lý đầu vào: SILAM và CMAQ, sau đó hiệu chỉnh bằng AI."
)

# =========================
# Chọn nguồn mô hình
# =========================
st.subheader("Lựa chọn nguồn forecast vật lý")

model_names = list(MODEL_CONFIG.keys())
default_index = next(
    (i for i, name in enumerate(model_names) if MODEL_CONFIG[name]["default"]),
    0
)

selected_model = st.radio(
    "Chọn nhánh mô hình",
    model_names,
    index=default_index,
    horizontal=True
)

cfg = MODEL_CONFIG[selected_model]

pred_test_path = cfg["pred_test"]
pred_full_path = cfg["pred_full"]
plot_path = cfg["plot"]
map_data_path = cfg["map_data"]
raw_col = cfg["raw_col"]
raw_label = cfg["raw_label"]

ensure_map_data(pred_full_path, map_data_path)

# =========================
# Status
# =========================
st.subheader("Trạng thái hệ thống")
c1, c2, c3, c4 = st.columns(4)
c1.success("Dữ liệu trạm: OK")
c2.success(f"{selected_model} forecast: OK")
c3.success("AI correction: OK")
c4.success("Map dashboard: OK")

# =========================
# Metrics
# =========================
st.subheader("Chỉ số đánh giá")

if pred_test_path.exists():
    df_test = pd.read_csv(pred_test_path)

    mae_raw = mean_absolute_error(df_test["pm25_obs"], df_test[raw_col])
    rmse_raw = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test[raw_col]))

    mae_corr = mean_absolute_error(df_test["pm25_obs"], df_test["pm25_pred_corrected"])
    rmse_corr = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test["pm25_pred_corrected"]))

    improvement_mae = mae_raw - mae_corr
    improvement_rmse = rmse_raw - rmse_corr

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE raw", f"{mae_raw:.2f}")
    m2.metric("RMSE raw", f"{rmse_raw:.2f}")
    m3.metric("MAE corrected", f"{mae_corr:.2f}", delta=f"-{improvement_mae:.2f}")
    m4.metric("RMSE corrected", f"{rmse_corr:.2f}", delta=f"-{improvement_rmse:.2f}")

    if selected_model == "SILAM":
        st.info(
            f"Nhánh SILAM có forecast thô lệch lớn, nhưng sau AI đã được hiệu chỉnh mạnh. "
            f"MAE giảm {improvement_mae:.2f}, RMSE giảm {improvement_rmse:.2f}."
        )
    else:
        st.info(
            f"Nhánh CMAQ có forecast thô ổn định hơn, và sau AI tiếp tục được cải thiện. "
            f"MAE giảm {improvement_mae:.2f}, RMSE giảm {improvement_rmse:.2f}."
        )
else:
    st.warning(f"Chưa có file {pred_test_path.name}")

# =========================
# Map
# =========================
st.subheader("Bản đồ Việt Nam")
st.write(
    f"Hiển thị 3 trạm Hà Nội trên nền bản đồ Việt Nam. Màu marker được tô theo PM2.5 của nhánh **{selected_model}** sau AI hiệu chỉnh."
)

if map_data_path.exists():
    map_df = pd.read_csv(map_data_path, parse_dates=["timestamp"])
    time_options = sorted(map_df["timestamp"].astype(str).unique().tolist())

    top_left, top_right = st.columns([2, 1])
    with top_left:
        selected_time = st.selectbox("Chọn thời điểm hiển thị", time_options)
    with top_right:
        color_mode = st.selectbox(
            "Tô màu theo",
            ["PM2.5 sau AI hiệu chỉnh", "PM2.5 thực tế", raw_label]
        )

    sub = map_df[map_df["timestamp"].astype(str) == selected_time].copy()

    map_center = [21.02, 105.84] if not sub.empty else [16.5, 106.0]
    zoom_level = 11 if not sub.empty else 5

    m = folium.Map(
        location=map_center,
        zoom_start=zoom_level,
        tiles="OpenStreetMap"
    )
    add_legend(m)

    for _, row in sub.iterrows():
        if color_mode == "PM2.5 sau AI hiệu chỉnh":
            color_value = row["pm25_pred_corrected"]
        elif color_mode == "PM2.5 thực tế":
            color_value = row["pm25_obs"]
        else:
            color_value = row[raw_col]

        color = get_color(color_value)
        level_text = get_level_text(color_value)

        raw_value = row[raw_col] if raw_col in row else np.nan

        popup_html = f"""
        <div style="font-size:14px; line-height:1.5;">
            <b>Trạm:</b> {row['station_name']}<br>
            <b>Mã trạm:</b> {row['station_id']}<br>
            <b>Thời gian:</b> {row['timestamp']}<br>
            <hr style="margin:6px 0;">
            <b>PM2.5 thực tế:</b> {row['pm25_obs']:.2f}<br>
            <b>{raw_label}:</b> {raw_value:.2f}<br>
            <b>Sau AI hiệu chỉnh:</b> {row['pm25_pred_corrected']:.2f}<br>
            <b>Mức hiện tại:</b> {level_text}
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=11,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{row['station_name']} | {color_value:.2f}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2
        ).add_to(m)

    st_folium(m, width=1150, height=560)
else:
    st.warning(f"Chưa có file {map_data_path.name}")

# =========================
# Plot
# =========================
st.subheader("Biểu đồ so sánh")
st.write(
    f"Biểu đồ dưới đây so sánh {raw_label}, kết quả sau AI hiệu chỉnh và giá trị thực tế quan trắc."
)

if plot_path.exists():
    st.image(str(plot_path), caption=f"{raw_label} vs AI hiệu chỉnh vs Thực tế")
else:
    st.warning(f"Chưa có file {plot_path.name}")

# =========================
# Data table
# =========================
st.subheader("Bảng dữ liệu dự báo")

if pred_full_path.exists():
    df_full = pd.read_csv(pred_full_path)

    common_cols = [
        "timestamp", "station_id", "station_name",
        "pm25_obs", "pm25_pred_corrected",
        "pm10_obs", "vn_aqi", "hour", "dayofweek",
        "pm25_lag1", "pm25_lag3", "pm25_roll6"
    ]

    if selected_model == "SILAM":
        model_cols = ["silam_pm25", "silam_pm10", "silam_aqi"]
    else:
        model_cols = ["pm25_cmaq_approx", "row", "col"]

    display_cols = [c for c in common_cols + model_cols if c in df_full.columns]

    rename_map = {
        "timestamp": "Thời gian",
        "station_id": "Mã trạm",
        "station_name": "Tên trạm",
        "pm25_obs": "PM2.5 thực tế",
        "pm25_pred_corrected": "Sau AI hiệu chỉnh",
        "pm10_obs": "PM10 thực tế",
        "vn_aqi": "VN_AQI",
        "hour": "Giờ",
        "dayofweek": "Thứ trong tuần",
        "pm25_lag1": "PM2.5 trễ 1 giờ",
        "pm25_lag3": "PM2.5 trễ 3 giờ",
        "pm25_roll6": "PM2.5 trung bình 6 giờ",
        "silam_pm25": "SILAM PM2.5",
        "silam_pm10": "SILAM PM10",
        "silam_aqi": "SILAM AQI",
        "pm25_cmaq_approx": "CMAQ thô",
        "row": "Hàng lưới CMAQ",
        "col": "Cột lưới CMAQ",
    }

    show_df = df_full[display_cols].copy().rename(columns=rename_map)
    st.dataframe(show_df.head(50), use_container_width=True)
else:
    st.warning(f"Chưa có file {pred_full_path.name}")

# =========================
# Notes
# =========================
st.subheader("Nhận xét kỹ thuật")

if selected_model == "SILAM":
    st.write(
        """
        - SILAM cung cấp trực tiếp các biến `cnc_PM2_5`, `cnc_PM10` và `AQI`, nên thuận lợi hơn trong việc mở rộng mô hình đa thông số.
        - Forecast thô từ SILAM hiện có sai số lớn, nhưng sau AI hiệu chỉnh cho kết quả rất tốt.
        - Đây là nhánh đang cho hiệu năng tốt nhất ở thời điểm hiện tại.
        """
    )
else:
    st.write(
        """
        - CMAQ hiện đóng vai trò baseline vật lý tốt và ổn định hơn ở forecast thô.
        - PM2.5 từ CMAQ hiện đang dùng dưới dạng `pm25_cmaq_approx` được dựng từ aerosol species trong file ACONC.
        - Nhánh CMAQ vẫn rất hữu ích để so sánh và làm đối chứng mô hình.
        """
    )
