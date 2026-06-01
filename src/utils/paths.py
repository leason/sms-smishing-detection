from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

RAW_MANUAL = RAW_DIR / "Dataset_5971.csv"
RAW_SYNTHETIC = RAW_DIR / "Dataset_10191.csv"

PROCESSED_MANUAL = PROCESSED_DIR / "manual_clean.csv"
PROCESSED_SYNTHETIC = PROCESSED_DIR / "synthetic_clean.csv"
PROCESSED_COMBINED = PROCESSED_DIR / "combined_clean.csv"
