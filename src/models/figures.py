"""Generate evaluation figures from metrics results."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

from src.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR

LABELS = ["ham", "spam", "smishing"]

EXPERIMENT_ORDER = ["manual_only", "synthetic_only", "combined"]


def plot_mean_metric_by_strategy(
    metrics_df: pd.DataFrame, metric: str, title: str, filename: str
) -> None:
    """Grouped bar chart of mean metric by model and training strategy.
    Also saves the source data as a sibling CSV for downstream plotting."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pivot = metrics_df.groupby(["model_name", "experiment_id"])[metric].mean().unstack()
    cols = [c for c in EXPERIMENT_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    ax = pivot.plot.bar(figsize=(11, 5), rot=0)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Training Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(bottom=max(0, pivot.min().min() - 0.05))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()

    # Sidecar CSV with the underlying data for downstream visualization.
    pivot.to_csv(FIGURES_DIR / filename.replace(".png", "_data.csv"))


def plot_confusion_matrix_best_model(
    manual: pd.DataFrame, synthetic: pd.DataFrame
) -> None:
    """Generate confusion matrix for best model on manual holdout."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from src.models.train import make_splits

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "best_model_metadata.json") as f:
        meta = json.load(f)

    splits = make_splits(manual, synthetic, seed=1)
    test_df = splits["manual_test"]
    y_true = test_df["label"]
    y_pred = model.predict(test_df)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Best Model ({meta['model_name']}) - Manual Holdout")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model_manual_holdout.png", dpi=150)
    plt.close()


def plot_smishing_f1_boxplot(metrics_df: pd.DataFrame) -> None:
    """Boxplot of smishing F1 by training strategy."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [c for c in EXPERIMENT_ORDER if c in metrics_df["experiment_id"].unique()]
    sns.boxplot(
        data=metrics_df, x="experiment_id", y="smishing_f1", hue="model_name",
        order=order, ax=ax,
    )
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Smishing F1")
    ax.set_title("Smishing F1 Distribution by Training Strategy")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "smishing_f1_boxplot_by_training_strategy.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    # Sidecar CSV: the raw per-seed values used for the boxplot.
    metrics_df[["seed", "model_name", "experiment_id", "smishing_f1"]].to_csv(
        FIGURES_DIR / "smishing_f1_boxplot_by_training_strategy_data.csv", index=False
    )


def generate_all_figures(
    metrics_df: pd.DataFrame,
    manual: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> None:
    """Generate all evaluation figures."""
    plot_mean_metric_by_strategy(
        metrics_df, "smishing_f1",
        "Mean Smishing F1 by Training Strategy",
        "mean_smishing_f1_by_training_strategy.png",
    )
    plot_mean_metric_by_strategy(
        metrics_df, "macro_f1",
        "Mean Macro F1 by Training Strategy",
        "mean_macro_f1_by_training_strategy.png",
    )
    plot_confusion_matrix_best_model(manual, synthetic)
    plot_smishing_f1_boxplot(metrics_df)
