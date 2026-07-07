"""Launcher to make PySide6 app importable regardless of working directory.

This module keeps the existing project structure.

Run:
    python app/app_run.py
or
    python -m app.app_run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_root_on_sys_path() -> None:
    # app_run.py -> app/ -> project root
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _ensure_root_on_sys_path()

    from app.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()

