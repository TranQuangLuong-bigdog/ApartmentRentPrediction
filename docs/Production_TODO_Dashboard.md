# Production TODO (Dashboard & UX)

Danh sách các phần còn thiếu để đạt đúng mục tiêu "production-ready".

- [ ] Gắn backend pipeline vào các trang Streamlit:
  - [ ] Train: gọi pipeline train, cập nhật progress bar, lưu model/history
  - [ ] Predict: load model, preprocess đúng, predict và export CSV
  - [ ] Evaluate: đọc metrics từ output/report và hiển thị charts
- [ ] Dataset Explorer:
  - [ ] Upload dataset, preview, pagination
  - [ ] Missing/outlier/correlation/hist/box/pair/distribution
- [ ] Dashboard:
  - [ ] Đọc metrics + dataset stats + model stats từ output/ và trained_models/
- [ ] Model Manager:
  - [ ] Lưu nhiều model (name/version)
  - [ ] Đổi tên/xóa/so sánh/đặt model mặc định
- [ ] Prediction History:
  - [ ] Lưu lịch sử prediction (file/SQLite ở bước sau)
  - [ ] Search/filter/export/xóa
- [ ] Settings:
  - [ ] Persist settings (file/json) và áp dụng vào pipeline
  - [ ] Dark/Light, language switch
- [ ] Exception Handling:
  - [ ] Chuẩn hoá Retry mechanism
  - [ ] Không crash toàn bộ app

