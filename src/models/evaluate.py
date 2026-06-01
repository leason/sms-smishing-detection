"""Model evaluation: metrics extraction and output generation."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.paths import METRICS_DIR, TABLES_DIR

LABELS = ["ham", "spam", "smishing"]


def extract_metrics(
    y_true,
    y_pred,
    seed: int,
    model_name: str,
    experiment_id: str,
    duplicate_mode: str,
    n_train: int,
    n_test: int,
) -> dict:
    """Compute all required metrics for a single run."""
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    return {
        "seed": seed,
        "model_name": model_name,
        "experiment_id": experiment_id,
        "duplicate_mode": duplicate_mode,
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "ham_precision": report["ham"]["precision"],
        "ham_recall": report["ham"]["recall"],
        "ham_f1": report["ham"]["f1-score"],
        "spam_precision": report["spam"]["precision"],
        "spam_recall": report["spam"]["recall"],
        "spam_f1": report["spam"]["f1-score"],
        "smishing_precision": report["smishing"]["precision"],
        "smishing_recall": report["smishing"]["recall"],
        "smishing_f1": report["smishing"]["f1-score"],
    }


def extract_confusion_matrix(
    y_true,
    y_pred,
    seed: int,
    model_name: str,
    experiment_id: str,
) -> list[dict]:
    """Return confusion matrix as list of long-format rows."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rows = []
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "experiment_id": experiment_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(cm[i, j]),
                }
            )
    return rows


def extract_classification_report(
    y_true,
    y_pred,
    seed: int,
    model_name: str,
    experiment_id: str,
) -> dict:
    """Return full classification report as a JSON-serializable dict."""
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "seed": seed,
        "model_name": model_name,
        "experiment_id": experiment_id,
        "report": report,
    }


def save_all_results(
    all_metrics: list[dict],
    all_cm: list[dict],
    all_reports: list[dict],
) -> None:
    """Save metrics CSV, confusion matrices CSV, and classification reports JSONL."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_metrics).to_csv(METRICS_DIR / "all_model_results.csv", index=False)
    pd.DataFrame(all_cm).to_csv(METRICS_DIR / "confusion_matrices.csv", index=False)

    with open(METRICS_DIR / "classification_reports.jsonl", "w") as f:
        for report in all_reports:
            f.write(json.dumps(report) + "\n")


def generate_summary_tables(metrics_df: pd.DataFrame) -> None:
    """Generate model and experiment performance summary tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        c for c in metrics_df.columns
        if c not in ("seed", "model_name", "experiment_id", "duplicate_mode", "n_train", "n_test")
    ]

    def _summarize(group_cols: list[str]) -> pd.DataFrame:
        grouped = metrics_df.groupby(group_cols)[metric_cols]
        summary = grouped.agg(["mean", "std", "median", "min", "max"])
        # add IQR
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        iqr = q3 - q1
        iqr.columns = pd.MultiIndex.from_product([iqr.columns, ["iqr"]])
        summary = summary.join(iqr)
        # sort columns by metric then stat
        summary = summary.sort_index(axis=1, level=0)
        return summary

    model_summary = _summarize(["model_name"])
    model_summary.to_csv(TABLES_DIR / "model_performance_summary.csv")

    experiment_summary = _summarize(["experiment_id"])
    experiment_summary.to_csv(TABLES_DIR / "experiment_performance_summary.csv")
