# Apartment Rent Prediction using Artificial Neural Network (ANN)

## Giới thiệu
Dự án này xây dựng mô hình **Artificial Neural Network (ANN)** để dự đoán **giá thuê hàng tháng** của căn hộ dựa trên các đặc trưng như diện tích, số phòng ngủ/phòng tắm, vị trí, tầng, nội thất, chỗ đậu xe và khoảng cách tới metro...

## Mục tiêu
- Xây dựng pipeline hoàn chỉnh: nạp dữ liệu → tiền xử lý → mã hóa → chuẩn hóa → chia train/test → huấn luyện ANN → đánh giá → tạo file dự đoán.
- Tổ chức code theo kiến trúc module để dễ mở rộng và kiểm thử.

## Kiến trúc thư mục
- `data/`: chứa dữ liệu thô (`raw/`) và dữ liệu đã xử lý (`processed/`)
- `notebook/`: notebook EDA / preprocessing / training / testing
- `src/`: toàn bộ logic chính (config, preprocessing, models, evaluation, api)
- `docs/`: tài liệu mô tả bài toán, dataset, workflow, kiến trúc ANN
- `trained_models/`: lưu model tốt nhất và history huấn luyện
- `output/`: lưu hình ảnh, báo cáo, kết quả dự đoán
- `tests/`: unit tests

## Hướng dẫn cài đặt
```bash
pip install -r requirements.txt
```

## Hướng dẫn chạy
### 1) Huấn luyện + đánh giá
```bash
python main.py
```

### 2) (Tuỳ chọn) API dự đoán
```bash
python -m src.api.predict_api
```

## Dataset
- Nguồn: Kaggle (Apartment Rent Dataset)
- Target column: `Rent`
- Feature: gồm cả số và phân loại → cần encoding và scaling.

## Workflow
1. Load dataset
2. Explore data
3. Handle missing values
4. Remove outliers
5. Feature engineering
6. Encoding (One-Hot)
7. Scaling (StandardScaler)
8. Train/Test split
9. Build ANN
10. Train
11. Evaluate
12. Save model
13. Predict

## Các chỉ số đánh giá
- **MAE**
- **MSE**
- **RMSE**
- **R2 Score**


