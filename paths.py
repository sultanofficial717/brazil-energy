"""
Path configuration helper for Brazilian Energy Forecasting repository.
Provides standardized directory access for data, figures, reports, and results.
"""
from scripts.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    get_data_path,
    get_figure_path,
    get_report_path,
    get_result_path
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "get_data_path",
    "get_figure_path",
    "get_report_path",
    "get_result_path"
]
