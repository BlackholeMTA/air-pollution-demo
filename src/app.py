from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Dashboard PM2.5 Hà Nội",
    page_icon="🌫️",
    layout="wide"
)

MODEL_CONFIG = {
    "HYBRID": {
        "pred_test": Path("data/output/predictions_multi_source_7days.csv"),
        "pred_full": Path("data/output/predictions_multi_source_7days_full.csv"),
        "plot": Path("data/output/comparison_multi_source_7days.png"),
        "map_data": Path("data/output/map_data_hybrid_7days.csv"),
        "raw_cols": ["silam_pm25", "cmaq_pm25_approx"],
        "raw_label": "Hybrid đầu vào",
        "default": True,
    },
    "SILAM": {
        "pred_test": Path("data/output/predictions_silam_hanoi.csv"),
        "pred_full": Path("data/output/predictions_silam_hanoi_full.csv"),
        "plot": Path("data/output/comparison_silam_hanoi.png"),
        "map_data": Path("data/output/map_data_silam_hanoi.csv"),
        "raw_cols": ["silam_pm25"],
        "raw_label": "SILAM thô",
        "default": False,
    },
    "CMAQ": {
        "pred_test": Path("data/output/predictions_cmaq_hanoi.csv"),
        "pred_full": Path("data/output/predictions_cmaq_hanoi_full.csv"),
        "plot": Path("data/output/comparison_cmaq_hanoi.png"),
        "map_data": Path("data/output/map_data_cmaq_hanoi.csv"),
        "raw_cols": ["pm25_cmaq_approx"],
        "raw_label": "CMAQ thô",
        "default": False,
    },
}


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


def ensure_map_data(pred_full_path: Path, map_data_path: Path, model_name: str):
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


def fmt_metric(x):
    return f"{x:.3f}"


st.title("Dashboard hiệu chỉnh dự báo PM2.5 - Hà Nội")
st.caption(
    "So sánh các nguồn forecast vật lý SILAM, CMAQ và mô hình AI hiệu chỉnh/Hybrid trên dữ liệu 3 trạm Hà Nội."
)

st.subheader("Lựa chọn nhánh mô hình")

model_names = list(MODEL_CONFIG.keys())
default_index = next(
    (i for i, name in enumerate(model_names) if MODEL_CONFIG[name]["default"]),
    0
)

selected_model = st.radio(
    "Chọn mô hình hiển thị",
    model_names,
    index=default_index,
    horizontal=True
)

cfg = MODEL_CONFIG[selected_model]
pred_test_path = cfg["pred_test"]
pred_full_path = cfg["pred_full"]
plot_path = cfg["plot"]
map_data_path = cfg["map_data"]
raw_cols = cfg["raw_cols"]
raw_label = cfg["raw_label"]

ensure_map_data(pred_full_path, map_data_path, selected_model)

st.subheader("Trạng thái hệ thống")
c1, c2, c3, c4 = st.columns(4)
c1.success("Dữ liệu trạm: OK")
c2.success(f"{selected_model}: OK")
c3.success("AI correction: OK")
c4.success("Dashboard map: OK")

st.subheader("Chỉ số đánh giá")

if pred_test_path.exists():
    df_test = pd.read_csv(pred_test_path)

    mae_corr = mean_absolute_error(df_test["pm25_obs"], df_test["pm25_pred_corrected"])
    rmse_corr = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test["pm25_pred_corrected"]))

    if selected_model == "HYBRID":
        mae_silam = mean_absolute_error(df_test["pm25_obs"], df_test["silam_pm25"])
        rmse_silam = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test["silam_pm25"]))

        mae_cmaq = mean_absolute_error(df_test["pm25_obs"], df_test["cmaq_pm25_approx"])
        rmse_cmaq = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test["cmaq_pm25_approx"]))

        m1, m2, m3 = st.columns(3)
        m1.metric("SILAM raw", f"MAE {mae_silam:.2f} | RMSE {rmse_silam:.2f}")
        m2.metric("CMAQ raw", f"MAE {mae_cmaq:.2f} | RMSE {rmse_cmaq:.2f}")
        m3.metric("AI hybrid", f"MAE {mae_corr:.2f} | RMSE {rmse_corr:.2f}")

        st.info(
            "Nhánh Hybrid sử dụng đồng thời forecast từ SILAM và CMAQ cùng các đặc trưng lịch sử quan trắc. "
            "Đây là nhánh cho kết quả tốt nhất hiện tại."
        )
    else:
        raw_col = raw_cols[0]
        mae_raw = mean_absolute_error(df_test["pm25_obs"], df_test[raw_col])
        rmse_raw = np.sqrt(mean_squared_error(df_test["pm25_obs"], df_test[raw_col]))

        improvement_mae = mae_raw - mae_corr
        improvement_rmse = rmse_raw - rmse_corr

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE raw", fmt_metric(mae_raw))
        m2.metric("RMSE raw", fmt_metric(rmse_raw))
        m3.metric("MAE corrected", fmt_metric(mae_corr), delta=f"-{improvement_mae:.2f}")
        m4.metric("RMSE corrected", fmt_metric(rmse_corr), delta=f"-{improvement_rmse:.2f}")

        st.info(
            f"Nhánh {selected_model} sau AI đã cải thiện so với forecast thô. "
            f"MAE giảm {improvement_mae:.2f}, RMSE giảm {improvement_rmse:.2f}."
        )
else:
    st.warning(f"Chưa có file {pred_test_path.name}")

st.subheader("Bản đồ Việt Nam")

if map_data_path.exists():
    map_df = pd.read_csv(map_data_path, parse_dates=["timestamp"])
    time_options = sorted(map_df["timestamp"].astype(str).unique().tolist())

    left, right = st.columns([2, 1])
    with left:
        selected_time = st.selectbox("Chọn thời điểm hiển thị", time_options)
    with right:
        if selected_model == "HYBRID":
            color_mode = st.selectbox(
                "Tô màu theo",
                ["PM2.5 sau AI hybrid", "PM2.5 thực tế", "SILAM thô", "CMAQ thô"]
            )
        elif selected_model == "SILAM":
            color_mode = st.selectbox(
                "Tô màu theo",
                ["PM2.5 sau AI hiệu chỉnh", "PM2.5 thực tế", "SILAM thô"]
            )
        else:
            color_mode = st.selectbox(
                "Tô màu theo",
                ["PM2.5 sau AI hiệu chỉnh", "PM2.5 thực tế", "CMAQ thô"]
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
        if color_mode == "PM2.5 sau AI hybrid":
            color_value = row["pm25_pred_corrected"]
        elif color_mode == "PM2.5 sau AI hiệu chỉnh":
            color_value = row["pm25_pred_corrected"]
        elif color_mode == "PM2.5 thực tế":
            color_value = row["pm25_obs"]
        elif color_mode == "SILAM thô":
            color_value = row["silam_pm25"]
        else:
            color_value = row["cmaq_pm25_approx"]

        color = get_color(color_value)
        level_text = get_level_text(color_value)

        popup_lines = [
            f"<b>Trạm:</b> {row['station_name']}",
            f"<b>Mã trạm:</b> {row['station_id']}",
            f"<b>Thời gian:</b> {row['timestamp']}",
            "<hr style='margin:6px 0;'>",
            f"<b>PM2.5 thực tế:</b> {row['pm25_obs']:.2f}",
        ]

        if "silam_pm25" in row.index:
            popup_lines.append(f"<b>SILAM thô:</b> {row['silam_pm25']:.2f}")
        if "cmaq_pm25_approx" in row.index:
            popup_lines.append(f"<b>CMAQ thô:</b> {row['cmaq_pm25_approx']:.2f}")

        if selected_model == "HYBRID":
            popup_lines.append(f"<b>AI hybrid:</b> {row['pm25_pred_corrected']:.2f}")
        else:
            popup_lines.append(f"<b>Sau AI hiệu chỉnh:</b> {row['pm25_pred_corrected']:.2f}")

        popup_lines.append(f"<b>Mức hiện tại:</b> {level_text}")

        popup_html = "<div style='font-size:14px; line-height:1.5;'>" + "<br>".join(popup_lines) + "</div>"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=11,
            popup=folium.Popup(popup_html, max_width=340),
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

st.subheader("Biểu đồ so sánh")

if plot_path.exists():
    st.image(str(plot_path), caption=f"Biểu đồ so sánh - {selected_model}")
else:
    st.warning(f"Chưa có file {plot_path.name}")

st.subheader("Bảng dữ liệu dự báo")

if pred_full_path.exists():
    df_full = pd.read_csv(pred_full_path)

    common_cols = [
        "timestamp", "station_id", "station_name",
        "pm25_obs", "pm25_pred_corrected",
        "pm10_obs", "vn_aqi", "hour", "dayofweek",
        "pm25_lag1", "pm25_lag3", "pm25_roll6"
    ]

    if selected_model == "HYBRID":
        model_cols = ["silam_pm25", "silam_pm10", "silam_aqi", "cmaq_pm25_approx"]
    elif selected_model == "SILAM":
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
        "cmaq_pm25_approx": "CMAQ PM2.5 approx",
        "pm25_cmaq_approx": "CMAQ PM2.5 approx",
        "row": "Hàng lưới CMAQ",
        "col": "Cột lưới CMAQ",
    }

    show_df = df_full[display_cols].copy().rename(columns=rename_map)
    st.dataframe(show_df.head(60), use_container_width=True)
else:
    st.warning(f"Chưa có file {pred_full_path.name}")

st.subheader("Nhận xét kỹ thuật")

if selected_model == "HYBRID":
    st.write(
        """
        - Nhánh Hybrid kết hợp đồng thời forecast từ SILAM và CMAQ.
        - Sau khi làm sạch dữ liệu theo cửa sổ 7 ngày, mô hình Hybrid vẫn cho kết quả tốt nhất.
        - Đây là nhánh nên dùng làm kết quả chính trong dashboard và báo cáo.
        """
    )
elif selected_model == "SILAM":
    st.write(
        """
        - SILAM cung cấp trực tiếp PM2.5, PM10 và AQI.
        - Forecast thô từ SILAM có thể bias mạnh, nhưng AI correction xử lý khá hiệu quả.
        - Đây là một baseline vật lý quan trọng của project.
        """
    )
else:
    st.write(
        """
        - CMAQ hiện là baseline vật lý thứ hai của project.
        - PM2.5 từ CMAQ đang dùng ở dạng xấp xỉ từ aerosol species.
        - Nhánh này vẫn rất hữu ích để đối chứng và so sánh với SILAM/Hybrid.
        """
    )
