# Hệ thống hiệu chỉnh dự báo PM2.5 bằng AI từ SILAM và CMAQ cho Hà Nội

## 1. Giới thiệu

Project này xây dựng một hệ thống thử nghiệm nhằm **hiệu chỉnh dự báo ô nhiễm không khí** bằng cách kết hợp giữa:

- **mô hình vật lý**: SILAM, CMAQ
- **mô hình học máy**: XGBoost

Mục tiêu chính là dự báo lại **PM2.5** tại các trạm quan trắc ở Hà Nội sao cho kết quả gần với số liệu thực tế hơn so với forecast thô từ mô hình vật lý.

Hệ thống hiện đã thực hiện được:

- đọc dữ liệu quan trắc theo giờ tại 3 trạm Hà Nội
- đọc forecast từ SILAM và CMAQ trong 7 ngày
- ánh xạ trạm vào lưới forecast
- xây dựng bộ dữ liệu huấn luyện đa nguồn
- huấn luyện mô hình AI hiệu chỉnh PM2.5
- so sánh kết quả giữa forecast thô và forecast sau AI
- trực quan hóa bằng dashboard web có bản đồ

---

## 2. Bài toán

Trong thực tế, các mô hình vật lý như **SILAM** và **CMAQ** có khả năng mô phỏng và dự báo chất lượng không khí trên lưới không gian, nhưng thường tồn tại **sai số hệ thống** so với quan trắc thực tế tại trạm.

Do đó, bài toán đặt ra là:

> Sử dụng mô hình học máy để học sai lệch giữa forecast vật lý và số liệu quan trắc, từ đó hiệu chỉnh forecast và tăng độ chính xác của dự báo PM2.5.

Cách tiếp cận này còn gọi là:

- **AI correction**
- **bias correction**
- **post-processing forecast**

---

## 3. Mục tiêu của project

- Xây dựng pipeline đọc dữ liệu forecast và dữ liệu trạm
- Kết hợp forecast từ **SILAM** và **CMAQ**
- Huấn luyện mô hình AI để hiệu chỉnh PM2.5
- So sánh các nhánh:
  - SILAM raw
  - CMAQ raw
  - AI Hybrid
- Trực quan hóa kết quả trên dashboard
- Tạo nền tảng để mở rộng sang:
  - PM10
  - AQI
  - O3
  - NO2 / NOx
  - SO2
  - CO

---

## 4. Dữ liệu sử dụng

### 4.1. Dữ liệu quan trắc trạm

Dữ liệu theo giờ của 3 trạm tại Hà Nội:

- 556 Nguyễn Văn Cừ
- Công viên Nhân Chính - Khuất Duy Tiến
- Số 1 Giải Phóng - Bạch Mai

Các cột quan trọng đang dùng gồm:

- `timestamp`
- `vn_aqi`
- `pm10_aqi`
- `pm25_aqi`
- `pm10_obs`
- `pm25_obs`
- `station_id`
- `station_name`

---

### 4.2. Dữ liệu SILAM

Dữ liệu SILAM được lưu theo thư mục ngày chạy mô hình:

```text
data/raw/silam/YYYYMMDD/
Trong mỗi thư mục có nhiều loại file, trong đó nhóm file quan trọng nhất cho pipeline hiện tại là:

PM_YYYYMMDDHH.nc4

Ví dụ:

PM_2025122500.nc4
PM_2025122501.nc4
...

Các biến được sử dụng từ SILAM:

cnc_PM2_5
cnc_PM10
AQI
Giải thích cấu trúc SILAM
Tên thư mục biểu diễn ngày chạy mô hình (run date)
Tên file biểu diễn thời điểm được dự báo (valid time)

Vì vậy, một thư mục ngày 25 có thể chứa cả forecast của ngày 26, 27... Điều này là bình thường đối với dữ liệu dự báo.

4.3. Dữ liệu CMAQ

Dữ liệu CMAQ hiện có 7 file ngày:

data/raw/cmaq/20251225/CCTM_ACONC_v532_gcc_v53_20251225.nc
data/raw/cmaq/20251226/CCTM_ACONC_v532_gcc_v53_20251226.nc
...
data/raw/cmaq/20251231/CCTM_ACONC_v532_gcc_v53_20251231.nc

Từ file CMAQ, project hiện trích xuất PM2.5 ở dạng:

cmaq_pm25_approx

Đây là PM2.5 xấp xỉ được dựng từ aerosol species trong file ACONC, vì bộ dữ liệu hiện tại chưa có PM25_TOT chuẩn từ workflow CMAQ đầy đủ.

4.4. Metadata trạm

File:

data/raw/stations/station_metadata.csv

Gồm:

station_id
station_name
lat
lon

File này được dùng để:

map trạm vào lưới SILAM theo lat/lon
map trạm vào lưới CMAQ theo row/col
5. Ý tưởng mô hình

Hệ thống sử dụng cách tiếp cận:

forecast vật lý + đặc trưng lịch sử + AI correction

Cụ thể:

SILAM và CMAQ cung cấp forecast nền
dữ liệu quan trắc lịch sử tại trạm cung cấp thông tin thực tế
XGBoost học mối quan hệ giữa forecast thô và quan trắc, từ đó dự báo lại PM2.5 thực tế
5.1. Các feature chính đang dùng
silam_pm25
silam_pm10
silam_aqi
cmaq_pm25_approx
pm25_lag1
pm25_lag3
pm25_lag6
pm25_roll6
pm25_roll12
pm10_obs
pm10_lag1
pm10_roll6
pm10_aqi
pm25_aqi
vn_aqi
hour
dayofweek
day
station_code
5.2. Biến mục tiêu
pm25_obs
5.3. Thuật toán
XGBoost Regressor
booster hiện tại: gbtree
6. Pipeline xử lý dữ liệu
Bước 1. Chuẩn hóa dữ liệu trạm

Dữ liệu quan trắc từ các file Excel được đọc, chuẩn hóa và ghép lại theo giờ.

Kết quả:

data/processed/hanoi_hourly_merged.csv
Bước 2. Trích forecast SILAM tại các trạm

Toàn bộ file PM_*.nc4 trong 7 ngày được quét, sau đó:

parse valid time từ tên file
chọn đúng forecast theo quy tắc thời gian
ánh xạ từng trạm vào lưới SILAM
lấy ra:
silam_pm25
silam_pm10
silam_aqi

Kết quả:

data/processed/silam_station_series_7days.csv
Bước 3. Trích forecast CMAQ tại các trạm

Từ 7 file ngày của CMAQ:

đọc 24 TSTEP cho mỗi ngày
tính cmaq_pm25_approx
lấy giá trị tại row, col của từng trạm

Kết quả:

data/processed/cmaq_station_series_7days.csv
Bước 4. Ghép đa nguồn

Ghép dữ liệu trạm, SILAM và CMAQ theo:

timestamp
station_id
station_name

Kết quả:

data/processed/train_multi_source_hanoi_7days.csv
Bước 5. Huấn luyện mô hình Hybrid

Mô hình XGBoost được huấn luyện trên bộ dữ liệu 7 ngày đã ghép.

Kết quả đầu ra:

data/output/xgb_multi_source_7days.pkl
data/output/predictions_multi_source_7days.csv
data/output/predictions_multi_source_7days_full.csv
Bước 6. Đánh giá mô hình

Mô hình được đánh giá bằng:

MAE
RMSE

và so sánh với:

SILAM raw
CMAQ raw
AI Hybrid
Bước 7. Trực quan hóa

Kết quả được hiển thị trên dashboard bằng:

Streamlit
Folium
biểu đồ Matplotlib

Dashboard hỗ trợ chọn các nhánh:

HYBRID
SILAM
CMAQ
7. Kết quả hiện tại
7.1. Bộ dữ liệu Hybrid 7 ngày
Số dòng: 504
Số trạm: 3
Khoảng thời gian:
2025-12-25 00:00:00
đến 2025-12-31 23:00:00

Điều này đúng với:

7 ngày × 24 giờ × 3 trạm = 504 dòng
7.2. Kết quả mô hình
SILAM raw
MAE: 102.928
RMSE: 116.868
CMAQ raw
MAE: 94.660
RMSE: 123.057
AI Hybrid
MAE: 4.996
RMSE: 6.922
7.3. Nhận xét kết quả

Kết quả cho thấy:

forecast thô từ SILAM và CMAQ đều có sai số lớn
mô hình AI Hybrid đã cải thiện rất mạnh chất lượng dự báo
AI Hybrid hiện là nhánh tốt nhất trong toàn bộ hệ thống

Có thể kết luận rằng:

Việc kết hợp đồng thời forecast từ SILAM và CMAQ cùng với đặc trưng lịch sử quan trắc giúp mô hình AI học tốt sai số hệ thống và cho kết quả vượt trội so với forecast thô.

8. Dashboard

Dashboard hiện hỗ trợ:

chọn nhánh mô hình:
HYBRID
SILAM
CMAQ
hiển thị:
chỉ số MAE/RMSE
bản đồ các trạm
biểu đồ so sánh
bảng dữ liệu dự báo

Nhánh HYBRID hiện được dùng làm nhánh chính vì cho hiệu năng tốt nhất.

9. Cấu trúc thư mục
air_pollution_demo/
├── data/
│   ├── raw/
│   │   ├── stations/
│   │   ├── silam/
│   │   └── cmaq/
│   ├── processed/
│   └── output/
├── db/
├── src/
│   ├── prepare_hanoi_hourly.py
│   ├── extract_silam_series_7days.py
│   ├── extract_cmaq_series_7days.py
│   ├── merge_multi_source_7days.py
│   ├── train_multi_source_7days.py
│   ├── evaluate_multi_source_7days.py
│   ├── plot_multi_source_7days.py
│   ├── prepare_map_data_hybrid.py
│   └── app.py
├── README.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
10. Cách chạy project
10.1. Build image Docker
docker build -t air-pollution-demo .
10.2. Trích dữ liệu SILAM 7 ngày
docker compose run --rm ai_demo python src/extract_silam_series_7days.py
10.3. Trích dữ liệu CMAQ 7 ngày
docker compose run --rm ai_demo python src/extract_cmaq_series_7days.py
10.4. Ghép dữ liệu đa nguồn
docker compose run --rm ai_demo python src/merge_multi_source_7days.py
10.5. Huấn luyện Hybrid model
docker compose run --rm ai_demo python src/train_multi_source_7days.py
docker compose run --rm ai_demo python src/evaluate_multi_source_7days.py
10.6. Tạo biểu đồ và dữ liệu bản đồ
docker compose run --rm ai_demo python src/plot_multi_source_7days.py
docker compose run --rm ai_demo python src/prepare_map_data_hybrid.py
10.7. Chạy dashboard
cd /home/admin_hoang/air_pollution_demo
docker compose run --rm -p 8501:8501 ai_demo streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501

Mở trình duyệt tại:

http://localhost:8501
11. Ưu điểm của hệ thống
Hướng tiếp cận đúng: kết hợp mô hình vật lý + AI
Đã xử lý được dữ liệu forecast thực tế nhiều ngày
Pipeline rõ ràng, tách biệt:
raw
processed
output
Hỗ trợ đồng thời 3 nhánh:
SILAM
CMAQ
Hybrid
Kết quả định lượng tốt, AI Hybrid vượt trội so với forecast thô
Có dashboard trực quan để trình bày và kiểm tra kết quả
12. Hạn chế hiện tại
Hiện mới tập trung vào PM2.5
Mới dùng 3 trạm Hà Nội
Dữ liệu mới ở mức 7 ngày
Nhánh CMAQ chưa dùng PM25_TOT chuẩn, mà đang dùng cmaq_pm25_approx
Chưa mở rộng sang:
PM10
AQI riêng
O3
NO2 / NOx
SO2
CO
Bản đồ hiện mới ở mức điểm trạm, chưa phải bản đồ lưới liên tục cho toàn miền Bắc
Hệ thống hiện là prototype nghiên cứu, chưa phải sản phẩm vận hành thời gian thực
13. Hướng phát triển
Mở rộng từ PM2.5 sang:
PM10
AQI
O3
NO2 / NOx
SO2
CO
Mở rộng thêm nhiều trạm ngoài Hà Nội
Tăng số ngày huấn luyện
So sánh nhiều mô hình hơn ngoài XGBoost
Tích hợp dữ liệu khí tượng và dữ liệu phụ trợ khác
Xây dựng bản đồ lưới cho phạm vi rộng hơn
Tiến tới hệ thống dự báo/cảnh báo ô nhiễm không khí quy mô lớn hơn
14. Kết luận

Project đã xây dựng thành công một pipeline hoàn chỉnh cho bài toán hiệu chỉnh dự báo PM2.5 bằng AI dựa trên hai nguồn forecast vật lý là SILAM và CMAQ. Trên bộ dữ liệu 7 ngày tại 3 trạm Hà Nội, mô hình Hybrid AI cho kết quả tốt nhất với:

MAE = 4.996
RMSE = 6.922

Kết quả này tốt hơn rất nhiều so với forecast thô từ cả SILAM và CMAQ, cho thấy tính khả thi và hiệu quả rõ rệt của hướng kết hợp mô hình vật lý và học máy.

Đây là nền tảng quan trọng để tiếp tục mở rộng thành hệ thống dự báo và cảnh báo ô nhiễm không khí ở quy mô lớn hơn trong tương lai.
