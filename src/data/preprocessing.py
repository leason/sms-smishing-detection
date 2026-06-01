"""Data loading, cleaning, and preprocessing for SMS smishing datasets."""

import re
import warnings
from typing import Literal

import pandas as pd

from src.utils.paths import (
    RAW_MANUAL,
    RAW_SYNTHETIC,
    PROCESSED_MANUAL,
    PROCESSED_SYNTHETIC,
    PROCESSED_COMBINED,
    PROCESSED_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS = {"ham", "spam", "smishing"}

LABEL_MAP = {
    "ham": "ham",
    "spam": "spam",
    "Spam": "spam",
    "smishing": "smishing",
    "Smishing": "smishing",
}

TRUTHY = {"yes", "y", "true", "1"}
FALSY = {"no", "n", "false", "0"}

COLUMN_RENAME = {
    "LABEL": "label",
    "TEXT": "text",
    "URL": "has_url",
    "EMAIL": "has_email",
    "PHONE": "has_phone",
}

INDICATOR_COLS = ["has_url", "has_email", "has_phone"]

DuplicateMode = Literal["keep_duplicates", "drop_exact_duplicates", "overlap_aware"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(path, dataset_source: str) -> pd.DataFrame:
    """Load a raw CSV and tag it with a dataset_source column."""
    df = pd.read_csv(path)
    df["dataset_source"] = dataset_source
    return df


def load_both_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw datasets, returning (manual_df, synthetic_df)."""
    manual = load_raw(RAW_MANUAL, "manual")
    synthetic = load_raw(RAW_SYNTHETIC, "synthetic")
    return manual, synthetic


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to standardized lowercase names."""
    return df.rename(columns=COLUMN_RENAME)


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw labels to normalized lowercase labels. Raises on unknown labels."""
    raw_labels = set(df["label"].unique())
    unknown = raw_labels - set(LABEL_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")
    df = df.copy()
    df["label"] = df["label"].map(LABEL_MAP)
    return df


def _normalize_indicator(value) -> int | None:
    """Convert a single indicator value to 0/1 or None."""
    s = str(value).strip().lower()
    if s in TRUTHY:
        return 1
    if s in FALSY:
        return 0
    warnings.warn(f"Unrecognized indicator value: {value!r}")
    return None


def normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize has_url, has_email, has_phone to binary integers.
    Unrecognized values are warned and filled with 0."""
    df = df.copy()
    for col in INDICATOR_COLS:
        df[col] = df[col].apply(_normalize_indicator)
        df[col] = df[col].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip and collapse whitespace. Preserves casing and URLs/emails/phones."""
    s = str(text).strip()
    return _MULTI_SPACE.sub(" ", s)


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add a text_clean column."""
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(df: pd.DataFrame, name: str = "dataset") -> None:
    """Run validation checks, raising on critical issues."""
    missing_text = df["text"].isna().sum()
    missing_label = df["label"].isna().sum()
    if missing_text > 0 or missing_label > 0:
        raise ValueError(
            f"{name}: found {missing_text} missing text, {missing_label} missing label"
        )
    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"{name}: unexpected labels after normalization: {bad_labels}")


# ---------------------------------------------------------------------------
# Duplicate / overlap handling
# ---------------------------------------------------------------------------


def handle_duplicates(
    manual: pd.DataFrame,
    synthetic: pd.DataFrame,
    mode: DuplicateMode = "drop_exact_duplicates",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply duplicate-handling strategy to both datasets.

    Modes:
        keep_duplicates: no changes.
        drop_exact_duplicates: drop dupes within each dataset on raw `text`.
        overlap_aware: drop_exact_duplicates + remove synthetic rows whose
            raw text appears in the manual dataset.
    """
    if mode == "keep_duplicates":
        return manual, synthetic

    # drop within-dataset duplicates on raw text
    manual = manual.drop_duplicates(subset=["text"], keep="first")
    synthetic = synthetic.drop_duplicates(subset=["text"], keep="first")

    if mode == "overlap_aware":
        manual_texts = set(manual["text"])
        synthetic = synthetic[~synthetic["text"].isin(manual_texts)]

    return manual, synthetic


def compute_overlap(manual: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of exact text matches between the two datasets."""
    manual_texts = set(manual["text"])
    overlap_mask = synthetic["text"].isin(manual_texts)
    overlap = synthetic.loc[overlap_mask, ["text", "label", "dataset_source"]].copy()
    overlap["in_manual"] = True
    return overlap


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "dataset_source",
    "label",
    "text",
    "text_clean",
    "has_url",
    "has_email",
    "has_phone",
]


def preprocess_dataset(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """Run the full normalization pipeline on a single raw DataFrame."""
    df = normalize_columns(df)
    df = normalize_labels(df)
    df = normalize_indicators(df)
    df = add_clean_text(df)
    validate(df, name)
    return df


def run_preprocessing(
    mode: DuplicateMode = "drop_exact_duplicates",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, deduplicate, and optionally save processed datasets.

    Returns (manual_clean, synthetic_clean, combined_clean).
    """
    manual_raw, synthetic_raw = load_both_raw()

    manual = preprocess_dataset(manual_raw, "manual")
    synthetic = preprocess_dataset(synthetic_raw, "synthetic")

    manual, synthetic = handle_duplicates(manual, synthetic, mode)

    combined = pd.concat([manual, synthetic], ignore_index=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        manual[OUTPUT_COLUMNS].to_csv(PROCESSED_MANUAL, index=False)
        synthetic[OUTPUT_COLUMNS].to_csv(PROCESSED_SYNTHETIC, index=False)
        combined[OUTPUT_COLUMNS].to_csv(PROCESSED_COMBINED, index=False)

    return manual, synthetic, combined
