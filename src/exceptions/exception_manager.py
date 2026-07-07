"""Exception manager để không để chương trình crash.

Giai đoạn hiện tại: skeleton, hiển thị thông điệp thân thiện và ghi log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import logging


@dataclass(frozen=True)
class FriendlyError:
    """Thông báo lỗi thân thiện."""

    title: str
    message: str


def classify_exception(exc: Exception) -> FriendlyError:
    """Phân loại exception để trả thông điệp dễ hiểu."""

    msg = str(exc)

    if "File" in exc.__class__.__name__ or "No such" in msg:
        return FriendlyError(
            title="Không tìm thấy file dữ liệu",
            message="Vui lòng kiểm tra bạn đã đặt đúng file CSV và chọn đúng đường dẫn/上传 file.",
        )

    if "encoding" in msg.lower():
        return FriendlyError(
            title="Lỗi mã hoá CSV",
            message="CSV có thể không đúng UTF-8. Hãy lưu lại file bằng UTF-8 rồi thử lại.",
        )

    if "tensorflow" in msg.lower():
        return FriendlyError(
            title="Lỗi TensorFlow",
            message="Có thể do môi trường TensorFlow hoặc thiếu tài nguyên (RAM/GPU). Hãy thử lại hoặc giảm kích thước dữ liệu/batch size.",
        )

    return FriendlyError(
        title="Đã xảy ra lỗi",
        message="Có lỗi xảy ra trong quá trình xử lý. Hãy bấm Retry để thử lại.",
    )


def handle_exception(exc: Exception, logger: Optional[logging.Logger] = None) -> FriendlyError:
    """Xử lý exception: ghi log và trả thông điệp thân thiện."""

    if logger:
        logger.exception("Exception occurred: %s", exc)

    return classify_exception(exc)

