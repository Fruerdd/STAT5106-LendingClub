"""Project paths derived from this file's location, not the working directory."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
CLEANED_DATA_DIR = PROJECT_DIR / "data" / "cleaned"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
