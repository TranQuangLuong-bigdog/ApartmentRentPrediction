"""Prompt templates cho AI assistant.

Giai đoạn hiện tại: không gọi LLM thực tế, chỉ sinh văn bản giải thích theo rules.
"""

from __future__ import annotations


EXPLAIN_MAE_TEMPLATE = (
    "MAE (Mean Absolute Error) đo mức sai trung bình giữa dự đoán và giá trị thực. "
    "MAE càng nhỏ thì mô hình dự đoán càng chính xác."
)

EXPLAIN_RMSE_TEMPLATE = (
    "RMSE (Root Mean Squared Error) cũng đo sai số, nhưng phạt mạnh các sai số lớn hơn. "
    "Nếu RMSE cao hơn MAE tương đối nhiều, có thể mô hình mắc lỗi lớn ở một số mẫu."
)

EXPLAIN_R2_TEMPLATE = (
    "R² cho biết tỉ lệ biến thiên của dữ liệu được mô hình giải thích. "
    "R² càng gần 1 thì mô hình càng phù hợp. "
    "R² khoảng 0 nghĩa là mô hình không tốt hơn trung bình đơn thuần."
)

