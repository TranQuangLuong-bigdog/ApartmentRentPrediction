"""PySide6 Desktop GUI entrypoint.

Chạy bằng:
    python app/app.py

GUI chỉ gọi service trong `src/`.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow

# Ensure UTF-8 for Vietnamese output/logs
os.environ.setdefault("PYTHONIOENCODING", "utf-8")





def main() -> None:
    """Entry point PySide6."""

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()



