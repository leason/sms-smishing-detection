"""Dataset provenance audit: overlap, duplicates, and leakage analysis."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import TABLES_DIR, FIGURES_DIR


def dataset_audit(manual: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset audit metrics. Returns a summary DataFrame."""
    rows = []
    for name, df in [("manual", manual), ("synthetic", synthetic)]:
        rows.append(
            {
                "dataset": name,
                "row_count": len(df),
                "ham_count": (df["label"] == "ham").sum(),
                "spam_count": (df["label"] == "spam").sum(),
                "smishing_count": (df["label"] == "smishing").sum(),
                "exact_duplicate_count": df.duplicated(subset=["text"], keep=False).sum(),
                "duplicate_pct": round(
                    df.duplicated(subset=["text"], keep=False).mean() * 100, 2
                ),
                "avg_message_length": round(df["text_clean"].str.len().mean(), 1),
                "vocabulary_size": len(
                    set(" ".join(df["text_clean"].fillna("")).lower().split())
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_dataset_overlap(
    manual: pd.DataFrame, synthetic: pd.DataFrame
) -> pd.DataFrame:
    """Find exact text matches between datasets."""
    manual_texts = set(manual["text"])
    overlap_rows = synthetic[synthetic["text"].isin(manual_texts)].copy()
    overlap_rows = overlap_rows[["text", "label"]].drop_duplicates()
    overlap_rows["in_manual"] = True
    return overlap_rows


def top_duplicate_messages(
    manual: pd.DataFrame, synthetic: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Return the most frequently duplicated messages across both datasets."""
    combined = pd.concat(
        [manual[["text", "label", "dataset_source"]], synthetic[["text", "label", "dataset_source"]]]
    )
    counts = combined.groupby("text").agg(
        count=("text", "size"),
        datasets=("dataset_source", lambda x: ",".join(sorted(set(x)))),
        label=("label", "first"),
    )
    return counts.sort_values("count", ascending=False).head(top_n).reset_index()


def leakage_audit(
    manual: pd.DataFrame, synthetic: pd.DataFrame
) -> pd.DataFrame:
    """Report overlap statistics that could cause train/test leakage."""
    manual_texts = set(manual["text"])
    synthetic_texts = set(synthetic["text"])
    overlap_texts = manual_texts & synthetic_texts

    rows = [
        {"metric": "manual_unique_texts", "value": len(manual_texts)},
        {"metric": "synthetic_unique_texts", "value": len(synthetic_texts)},
        {"metric": "overlapping_texts", "value": len(overlap_texts)},
        {
            "metric": "overlap_pct_of_manual",
            "value": round(len(overlap_texts) / len(manual_texts) * 100, 2)
            if manual_texts
            else 0,
        },
        {
            "metric": "overlap_pct_of_synthetic",
            "value": round(len(overlap_texts) / len(synthetic_texts) * 100, 2)
            if synthetic_texts
            else 0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_class_balance(manual: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    """Bar chart comparing class distributions."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, df) in zip(axes, [("Manual", manual), ("Synthetic", synthetic)]):
        counts = df["label"].value_counts().reindex(["ham", "spam", "smishing"])
        counts.plot.bar(ax=ax, color=["steelblue", "orange", "firebrick"])
        ax.set_title(f"{name} Dataset")
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance_comparison.png", dpi=150)
    plt.close()


def plot_duplicate_rates(audit_df: pd.DataFrame) -> None:
    """Bar chart of duplicate percentages."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(audit_df["dataset"], audit_df["duplicate_pct"], color=["steelblue", "orange"])
    ax.set_ylabel("Duplicate %")
    ax.set_title("Duplicate Rate by Dataset")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duplicate_rate_comparison.png", dpi=150)
    plt.close()


def plot_overlap_summary(leakage_df: pd.DataFrame) -> None:
    """Simple bar chart of overlap metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pct_rows = leakage_df[leakage_df["metric"].str.contains("pct")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pct_rows["metric"], pct_rows["value"], color=["steelblue", "orange"])
    ax.set_ylabel("Overlap %")
    ax.set_title("Cross-Dataset Text Overlap")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_overlap_summary.png", dpi=150)
    plt.close()


def plot_message_lengths(manual: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    """Histogram of message lengths by dataset."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(manual["text_clean"].str.len(), bins=50, alpha=0.6, label="Manual", color="steelblue")
    ax.hist(synthetic["text_clean"].str.len(), bins=50, alpha=0.6, label="Synthetic", color="orange")
    ax.set_xlabel("Message Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Message Length Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "message_length_distribution.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Run all audits
# ---------------------------------------------------------------------------


def run_audit(
    manual: pd.DataFrame, synthetic: pd.DataFrame, save: bool = True
) -> dict:
    """Run all audit steps. Returns dict of DataFrames."""
    audit_df = dataset_audit(manual, synthetic)
    overlap_df = cross_dataset_overlap(manual, synthetic)
    top_dupes_df = top_duplicate_messages(manual, synthetic)
    leakage_df = leakage_audit(manual, synthetic)

    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(TABLES_DIR / "dataset_audit.csv", index=False)
        overlap_df.to_csv(TABLES_DIR / "cross_dataset_overlap.csv", index=False)
        top_dupes_df.to_csv(TABLES_DIR / "top_duplicate_messages.csv", index=False)
        leakage_df.to_csv(TABLES_DIR / "leakage_audit.csv", index=False)

        plot_class_balance(manual, synthetic)
        plot_duplicate_rates(audit_df)
        plot_overlap_summary(leakage_df)
        plot_message_lengths(manual, synthetic)

    return {
        "audit": audit_df,
        "overlap": overlap_df,
        "top_duplicates": top_dupes_df,
        "leakage": leakage_df,
    }
