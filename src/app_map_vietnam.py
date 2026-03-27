import streamlit as st
import pandas as pd
import folium
from pathlib import Path
from streamlit_folium import st_folium

MAP_FILE = Path("data/output/map_data_cmaq_hanoi.csv")

st.set_page_config(page_title="Bản đồ Việt Nam - PM2.5", layout="wide")

st.title("Bản đồ Việt Nam - dự báo PM2.5 hiệu chỉnh")
st.caption("Hiển thị 3 trạm Hà Nội trên nền bản đồ Việt Nam")

if not MAP_FILE.exists():
    st.error("Chưa có file map_data_cmaq_hanoi.csv")
    st.stop()

df = pd.read_csv(MAP_FILE, parse_dates=["timestamp"])

# chọn thời gian
time_options = sorted(df["timestamp"].astype(str).unique().tolist())
selected_time = st.selectbox("Chọn thời điểm", time_options)

sub = df[df["timestamp"].astype(str) == selected_time].copy()

m = folium.Map(location=[16.5, 106.0], zoom_start=5, tiles="OpenStreetMap")

def get_color(pm25):
    if pm25 <= 12:
        return "green"
    elif pm25 <= 35:
        return "orange"
    elif pm25 <= 55:
        return "red"
    else:
        return "darkred"

for _, row in sub.iterrows():
    color = get_color(row["pm25_pred_corrected"])

    popup_html = f"""
    <b>Trạm:</b> {row['station_name']}<br>
    <b>PM2.5 thực tế:</b> {row['pm25_obs']:.2f}<br>
    <b>CMAQ thô:</b> {row['pm25_cmaq_approx']:.2f}<br>
    <b>Sau AI hiệu chỉnh:</b> {row['pm25_pred_corrected']:.2f}
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=10,
        popup=folium.Popup(popup_html, max_width=300),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=row["station_name"]
    ).add_to(m)

st.subheader("Bản đồ trạm")
st_folium(m, width=1100, height=600)

st.subheader("Bảng dữ liệu tại thời điểm đã chọn")
st.dataframe(
    sub.rename(columns={
        "timestamp": "Thời gian",
        "station_id": "Mã trạm",
        "station_name": "Tên trạm",
        "lat": "Vĩ độ",
        "lon": "Kinh độ",
        "pm25_obs": "PM2.5 thực tế",
        "pm25_cmaq_approx": "CMAQ thô",
        "pm25_pred_corrected": "Sau AI hiệu chỉnh"
    }),
    use_container_width=True
)
