# Hệ thống hiệu chỉnh dự báo PM2.5 bằng AI từ CMAQ/SILAM cho Hà Nội

## 1. Giới thiệu

Project này xây dựng một pipeline học máy để hiệu chỉnh dự báo ô nhiễm không khí từ các mô hình vật lý như CMAQ và SILAM.  
Mục tiêu hiện tại là hiệu chỉnh dự báo **PM2.5** tại 3 trạm quan trắc ở Hà Nội bằng mô hình **XGBoost**, sau đó trực quan hóa kết quả trên dashboard web có bản đồ.

Đầu vào của hệ thống gồm:
- dữ liệu quan trắc theo giờ tại các trạm Hà Nội,
- forecast từ mô hình vật lý CMAQ,
- forecast từ mô hình vật lý SILAM.

Đầu ra của hệ thống gồm:
- dự báo PM2.5 sau AI hiệu chỉnh,
- chỉ số đánh giá mô hình,
- dashboard hiển thị bản đồ, biểu đồ và bảng dữ liệu.

---

## 2. Mục tiêu

- Xây dựng pipeline đọc dữ liệu quan trắc và forecast.
- Map vị trí trạm quan trắc vào lưới forecast.
- Huấn luyện mô hình AI để hiệu chỉnh dự báo PM2.5.
- So sánh hiệu quả giữa 2 nhánh forecast đầu vào: **CMAQ** và **SILAM**.
- Hiển thị kết quả trên dashboard web có bản đồ.

---

## 3. Dữ liệu sử dụng

### 3.1. Dữ liệu trạm quan trắc
Dữ liệu theo giờ của 3 trạm tại Hà Nội:
- 556 Nguyễn Văn Cừ
- Công viên Nhân Chính - Khuất Duy Tiến
- Số 1 Giải Phóng - Bạch Mai

Các cột sử dụng gồm:
- Datetime
- VN_AQI
- PM-10
- PM-2-5
- và một số thông số khác tùy trạm

### 3.2. Dữ liệu CMAQ
File:
- `data/raw/cmaq/CCTM_ACONC_v532_gcc_v53_20251225.nc`

Ghi chú:
- file hiện tại không chứa trực tiếp `PM25_TOT`
- PM2.5 từ CMAQ đang được xây dựng dưới dạng **xấp xỉ** từ aerosol species
- tên biến sử dụng trong project: `pm25_cmaq_approx`

### 3.3. Dữ liệu SILAM
Các file trong thư mục:
- `data/raw/silam/20251227/PM_*.nc4`

Các biến dùng từ SILAM:
- `cnc_PM2_5`
- `cnc_PM10`
- `AQI`

### 3.4. Metadata trạm
File:
- `data/raw/stations/station_metadata.csv`

Gồm:
- `station_id`
- `station_name`
- `lat`
- `lon`

---

## 4. Ý tưởng mô hình

Project sử dụng cách tiếp cận **AI correction / bias correction**:

- forecast thô từ mô hình vật lý (CMAQ hoặc SILAM) được dùng làm đầu vào nền,
- dữ liệu lịch sử tại trạm được dùng làm feature bổ sung,
- mô hình XGBoost học cách hiệu chỉnh forecast để gần với quan trắc thực tế hơn.

### 4.1. Đầu vào mô hình
Ví dụ các feature đang dùng:
- forecast PM2.5 từ mô hình vật lý
- PM2.5 trễ 1 giờ
- PM2.5 trễ 3 giờ
- PM2.5 trung bình 6 giờ
- PM10 quan trắc
- PM10 trễ 1 giờ
- PM10 trung bình 6 giờ
- PM10 AQI
- PM2.5 AQI
- VN_AQI
- hour
- dayofweek
- station_code

### 4.2. Đầu ra mô hình
- `pm25_pred_corrected`

### 4.3. Thuật toán
- `XGBoost Regressor`
- booster hiện tại: `gbtree`

---

## 5. Pipeline xử lý

### 5.1. Chuẩn hóa dữ liệu trạm
File chính:
- `src/prepare_hanoi_hourly.py`

Kết quả:
- `data/processed/hanoi_hourly_merged.csv`

### 5.2. Nhánh CMAQ
- đọc file CMAQ
- trích PM2.5 xấp xỉ
- map trạm vào lưới CMAQ
- ghép forecast với dữ liệu trạm
- train model

Các file chính:
- `src/extract_cmaq_pm25.py`
- `src/map_stations_to_cmaq.py`
- `src/extract_cmaq_at_stations.py`
- `src/merge_station_cmaq_day.py`
- `src/train_cmaq_hanoi.py`

### 5.3. Nhánh SILAM
- đọc chuỗi file `PM_*.nc4`
- trích forecast tại 3 trạm
- ghép với dữ liệu trạm
- train model

Các file chính:
- `src/extract_silam_series_at_stations.py`
- `src/merge_station_silam_series.py`
- `src/train_silam_hanoi_series.py`

### 5.4. Đánh giá
Các chỉ số:
- MAE
- RMSE

Các file chính:
- `src/evaluate_cmaq_vs_ai.py`
- `src/evaluate_silam_vs_ai.py`

### 5.5. Trực quan hóa
- dashboard Streamlit
- bản đồ Folium
- biểu đồ so sánh forecast thô vs AI vs thực tế

File chính:
- `src/app.py`

---

## 6. Kết quả hiện tại

### 6.1. Nhánh CMAQ
- MAE raw: **15.074**
- RMSE raw: **18.641**
- MAE corrected: **9.724**
- RMSE corrected: **10.607**

Mức cải thiện:
- MAE giảm: **5.350**
- RMSE giảm: **8.035**

### 6.2. Nhánh SILAM
- MAE raw: **120.166**
- RMSE raw: **123.872**
- MAE corrected: **5.463**
- RMSE corrected: **6.758**

Mức cải thiện:
- MAE giảm: **114.703**
- RMSE giảm: **117.115**

### 6.3. Nhận xét
- Forecast thô từ **CMAQ** ổn định hơn forecast thô từ **SILAM**
- Tuy nhiên sau AI hiệu chỉnh, nhánh **SILAM + AI** cho kết quả tốt hơn trong thử nghiệm hiện tại
- Nhánh **SILAM** đang được xem là nhánh mô hình chính ở thời điểm hiện tại
- Nhánh **CMAQ** được giữ lại như một baseline vật lý quan trọng để so sánh

---

## 7. Dashboard

Dashboard hiện hỗ trợ:
- chọn nhánh mô hình: `SILAM` hoặc `CMAQ`
- hiển thị chỉ số MAE/RMSE
- hiển thị bản đồ các trạm trên nền bản đồ Việt Nam
- hiển thị biểu đồ so sánh
- hiển thị bảng dữ liệu dự báo

---

## 8. Cấu trúc thư mục chính

```text
air_pollution_demo/
├── data/
│   ├── raw/
│   │   ├── cmaq/
│   │   ├── silam/
│   │   └── stations/
│   ├── processed/
│   └── output/
├── db/
├── src/
│   ├── prepare_hanoi_hourly.py
│   ├── extract_cmaq_pm25.py
│   ├── map_stations_to_cmaq.py
│   ├── extract_cmaq_at_stations.py
│   ├── merge_station_cmaq_day.py
│   ├── train_cmaq_hanoi.py
│   ├── evaluate_cmaq_vs_ai.py
│   ├── extract_silam_series_at_stations.py
│   ├── merge_station_silam_series.py
│   ├── train_silam_hanoi_series.py
│   ├── evaluate_silam_vs_ai.py
│   └── app.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt

Cách chạy project
9. Cách chạy project
9.1. Build image Docker
docker build -t air-pollution-demo .
9.2. Huấn luyện nhánh CMAQ
docker compose run --rm ai_demo python src/train_cmaq_hanoi.py
docker compose run --rm ai_demo python src/evaluate_cmaq_vs_ai.py
docker compose run --rm ai_demo python src/plot_cmaq_results.py
docker compose run --rm ai_demo python src/prepare_map_data.py
9.3. Huấn luyện nhánh SILAM
docker compose run --rm ai_demo python src/train_silam_hanoi_series.py
docker compose run --rm ai_demo python src/evaluate_silam_vs_ai.py
docker compose run --rm ai_demo python src/plot_silam_results.py
docker compose run --rm ai_demo python src/prepare_map_data_silam.py
9.4. Chạy dashboard
cd /home/admin_hoang/air_pollution_demo
docker compose run --rm -p 8501:8501 ai_demo streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501

Sau đó mở trình duyệt tại:

http://localhost:8501
9.5. Một số lưu ý khi chạy
Lệnh docker compose phải chạy trong thư mục chứa docker-compose.yml
Các file *:Zone.Identifier là file phụ do Windows tạo, không dùng trong pipeline
Dashboard hiện hỗ trợ chuyển đổi giữa 2 nhánh forecast vật lý:
SILAM
CMAQ
10. Hạn chế hiện tại
Project hiện mới tập trung vào PM2.5
Dữ liệu thử nghiệm hiện mới gồm 3 trạm tại Hà Nội
Nhánh CMAQ hiện chưa có PM25_TOT chuẩn từ workflow CMAQ đầy đủ; đầu vào đang dùng là pm25_cmaq_approx được dựng từ aerosol species
Nhánh SILAM hiện có forecast PM2.5, PM10 và AQI trực tiếp, nhưng mới được thử nghiệm trên tập thời gian còn hạn chế
Quy mô mô hình hiện tại vẫn là prototype nghiên cứu, chưa phải hệ thống dự báo vận hành cho toàn miền Bắc
Chưa mở rộng sang các thông số:
PM10
O3
NO2 / NOx
SO2
CO
AQI hiệu chỉnh riêng
Chưa tích hợp thêm các nguồn dữ liệu bổ sung như:
khí tượng chi tiết
viễn thám
phát thải
Chưa xây dựng bản đồ phân bố theo lưới cho toàn miền Bắc, hiện mới hiển thị các điểm trạm trên nền bản đồ
Chưa triển khai cơ chế cập nhật forecast tự động theo thời gian thực
11. Hướng phát triển
Mở rộng từ PM2.5 sang PM10 và AQI
Bổ sung thêm nhiều trạm ngoài Hà Nội
Mở rộng thêm nhiều ngày forecast từ SILAM/CMAQ
Tích hợp thêm các thông số khí như NO2, SO2, CO, O3 vào feature
Mở rộng dashboard lên bản đồ dự báo quy mô lớn hơn
Tiến tới hệ thống dự báo/cảnh báo 48h cho nhiều thông số ô nhiễm
