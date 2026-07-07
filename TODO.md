# TODO - ApartmentRentPrediction (ANN + PySide6 GUI)

- [ ] Bước 1: Sửa `app/app.py` thành PySide6 entrypoint (mở `MainWindow`).
- [x] Bước 2: Hoàn thiện router/menu trong `app/main_window.py` và `app/sidebar.py` để chuyển page.

- [x] Bước 3: Viết PySide6 UI cho các page: 
  - [x] `app/pages/dataset_page.py`
  - [x] `app/pages/training_page.py`
  - [x] `app/pages/prediction_page.py`
  - [x] `app/pages/evaluation_page.py`
  - [x] `app/pages/model_page.py`
  - [x] `app/pages/settings_page.py`

- [x] Bước 4: Mỗi page chỉ gọi service trong `src/` (không chứa logic AI thô).

- [x] Bước 5: Cập nhật `requirements.txt` thêm `PySide6` (giữ `streamlit` cho an toàn).

- [ ] Bước 6: Chạy smoke test:
  - [ ] `python main.py` (đảm bảo train/evaluate CLI vẫn chạy)
  - [ ] `python app/app.py` (đảm bảo GUI mở và chuyển page)

- [ ] Bước 7: UI bổ sung upload ảnh + demo sample_properties.csv
  - [ ] DatasetPage: import/preview/stats/missing/duplicate/upload image + sort
  - [ ] PredictionPage: upload image + input fields + lưu history

- [ ] Bước 8: ModelPage đầy đủ CRUD (load/delete/rename/activate)


