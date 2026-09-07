import os
from pathlib import Path

# Project root is the parent directory of 'scripts/'
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'scripts' else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [DATA_DIR, FIGURES_DIR, REPORTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_data_path(filename: str) -> str:
    """Finds an existing data file in data/ or root directory, or returns default data/ path."""
    for candidate in [DATA_DIR / filename, Path.cwd() / "data" / filename, Path.cwd() / filename, Path(filename)]:
        if candidate.exists():
            return str(candidate.resolve())
    return str((DATA_DIR / filename).resolve())

def get_figure_path(filename: str) -> str:
    """Returns absolute path to save or load a figure in figures/."""
    return str((FIGURES_DIR / filename).resolve())

def get_report_path(filename: str) -> str:
    """Returns absolute path to save or load a report in reports/."""
    return str((REPORTS_DIR / filename).resolve())

def get_result_path(filename: str) -> str:
    """Returns absolute path to save or load benchmark results in results/."""
    return str((RESULTS_DIR / filename).resolve())
