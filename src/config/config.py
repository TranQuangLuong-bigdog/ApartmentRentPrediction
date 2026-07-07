"""Cấu hình đường dẫn dự án sử dụng pathlib.

Module này cung cấp các đường dẫn tương đối dựa trên vị trí file hiện tại.
Không hard-code đường dẫn tuyệt đối.

Yêu cầu:
- Mọi đường dẫn phải là relative trong project.
- Dùng pathlib để đảm bảo tính tương thích hệ thống.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Đối tượng chứa các đường dẫn dự án."""

    root: Path
    data_raw: Path
    data_processed: Path
    data_sample_input: Path
    artifacts_dir: Path
    trained_models: Path
    output_dir: Path
    output_figures: Path
    output_reports: Path
    output_prediction: Path
    report_dir: Path
    logs_dir: Path


def get_project_paths() -> ProjectPaths:
    """Lấy các đường dẫn dự án.

    Returns:
        ProjectPaths: Các đường dẫn chính của dự án.
    """

    # src/config/config.py -> src/config -> src -> project_root
    root = Path(__file__).resolve().parents[2]

    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    data_sample_input = root / "data" / "sample_input"

    artifacts_dir = root / "trained_models"
    trained_models = artifacts_dir

    output_dir = root / "output"
    output_figures = output_dir / "figures"
    output_reports = output_dir / "reports"
    output_prediction = output_dir / "prediction"

    report_dir = root / "report"
    logs_dir = root / "output" / "logs"

    return ProjectPaths(
        root=root,
        data_raw=data_raw,
        data_processed=data_processed,
        data_sample_input=data_sample_input,
        artifacts_dir=artifacts_dir,
        trained_models=trained_models,
        output_dir=output_dir,
        output_figures=output_figures,
        output_reports=output_reports,
        output_prediction=output_prediction,
        report_dir=report_dir,
        logs_dir=logs_dir,
    )


def ensure_directories(paths: ProjectPaths) -> None:
    """Tạo các thư mục cần thiết nếu chưa tồn tại.

    Args:
        paths: ProjectPaths.
    """

    for p in [
        paths.data_raw,
        paths.data_processed,
        paths.data_sample_input,
        paths.trained_models,
        paths.output_figures,
        paths.output_reports,
        paths.output_prediction,
        paths.report_dir,
        paths.logs_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

