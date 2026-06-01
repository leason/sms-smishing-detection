"""Model training: experiment loop across seeds, models, and training strategies."""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.features.text_features import build_pipeline
from src.models.evaluate import (
    extract_classification_report,
    extract_confusion_matrix,
    extract_metrics,
    generate_summary_tables,
    save_all_results,
)
from src.utils.paths import MODELS_DIR

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    "MultinomialNB": lambda seed: MultinomialNB(),
    "LogisticRegression": lambda seed: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    ),
    "LinearSVC": lambda seed: CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000)
    ),
}

SEEDS = list(range(1, 31))
EXPERIMENTS = ["manual_only", "synthetic_only", "combined"]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def make_splits(
    manual: pd.DataFrame, synthetic: pd.DataFrame, seed: int
) -> dict:
    """Create stratified 80/20 splits and size-matched synthetic sample.

    Returns dict with keys:
        manual_train, manual_test, synthetic_train, synthetic_train_matched, combined_train
    """
    manual_train, manual_test = train_test_split(
        manual, test_size=0.20, stratify=manual["label"], random_state=seed
    )
    synthetic_train, _synthetic_test = train_test_split(
        synthetic, test_size=0.20, stratify=synthetic["label"], random_state=seed
    )

    # Size-match: stratified downsample of synthetic to match manual train size
    n_manual_train = len(manual_train)
    if len(synthetic_train) > n_manual_train:
        # Proportional sample per class
        sample_indices = []
        for label, group in synthetic_train.groupby("label"):
            n_sample = max(1, int(len(group) * n_manual_train / len(synthetic_train)))
            sample_indices.extend(group.sample(n=n_sample, random_state=seed).index.tolist())
        synthetic_train_matched = synthetic_train.loc[sample_indices]

        # Adjust to exactly match n_manual_train if rounding caused a mismatch
        diff = n_manual_train - len(synthetic_train_matched)
        if diff > 0:
            remaining = synthetic_train.drop(sample_indices)
            extra = remaining.sample(n=diff, random_state=seed)
            synthetic_train_matched = pd.concat([synthetic_train_matched, extra])
        elif diff < 0:
            synthetic_train_matched = synthetic_train_matched.sample(
                n=n_manual_train, random_state=seed
            )
    else:
        synthetic_train_matched = synthetic_train

    combined_train = pd.concat([manual_train, synthetic_train], ignore_index=True)

    return {
        "manual_train": manual_train,
        "manual_test": manual_test,
        "synthetic_train": synthetic_train,
        "synthetic_train_matched": synthetic_train_matched,
        "combined_train": combined_train,
    }


def _get_experiment_data(
    splits: dict, experiment_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for a given experiment.
    All experiments test against manual holdout."""
    test = splits["manual_test"]
    if experiment_id == "manual_only":
        return splits["manual_train"], test
    elif experiment_id == "synthetic_only":
        return splits["synthetic_train_matched"], test
    elif experiment_id == "combined":
        return splits["combined_train"], test
    else:
        raise ValueError(f"Unknown experiment: {experiment_id}")


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_experiments(
    manual: pd.DataFrame,
    synthetic: pd.DataFrame,
    duplicate_mode: str = "drop_exact_duplicates",
    seeds: list[int] | None = None,
    models: dict | None = None,
    experiments: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full experiment matrix and save all outputs.

    Returns the metrics DataFrame.
    """
    seeds = seeds or SEEDS
    models = models or MODELS
    experiments = experiments or EXPERIMENTS

    all_metrics = []
    all_cm = []
    all_reports = []

    total = len(seeds) * len(models) * len(experiments)
    run_num = 0

    for seed in seeds:
        splits = make_splits(manual, synthetic, seed)

        for model_name, model_factory in models.items():
            for experiment_id in experiments:
                run_num += 1
                if verbose:
                    print(f"  [{run_num}/{total}] seed={seed} model={model_name} exp={experiment_id}")

                train_df, test_df = _get_experiment_data(splits, experiment_id)

                pipeline = build_pipeline(model_factory(seed))
                pipeline.fit(train_df, train_df["label"])

                y_pred = pipeline.predict(test_df)
                y_true = test_df["label"]

                metrics = extract_metrics(
                    y_true, y_pred, seed, model_name, experiment_id,
                    duplicate_mode, len(train_df), len(test_df),
                )
                all_metrics.append(metrics)
                all_cm.extend(extract_confusion_matrix(y_true, y_pred, seed, model_name, experiment_id))
                all_reports.append(extract_classification_report(y_true, y_pred, seed, model_name, experiment_id))

    # Save outputs
    save_all_results(all_metrics, all_cm, all_reports)

    metrics_df = pd.DataFrame(all_metrics)
    generate_summary_tables(metrics_df)

    # Select and save best model
    _select_and_save_best_model(manual, synthetic, metrics_df, models)

    return metrics_df


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------


def _select_and_save_best_model(
    manual: pd.DataFrame,
    synthetic: pd.DataFrame,
    metrics_df: pd.DataFrame,
    models: dict,
) -> None:
    """Select the best model/experiment combo and save it."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Rank by mean smishing F1, then macro F1
    summary = (
        metrics_df.groupby(["model_name", "experiment_id"])
        .agg(mean_smishing_f1=("smishing_f1", "mean"), mean_macro_f1=("macro_f1", "mean"))
        .sort_values(["mean_smishing_f1", "mean_macro_f1"], ascending=False)
    )

    best = summary.iloc[0]
    best_model_name = summary.index[0][0]
    best_experiment = summary.index[0][1]

    # Retrain on full training data for the best experiment using seed=1
    splits = make_splits(manual, synthetic, seed=1)
    train_df, _ = _get_experiment_data(splits, best_experiment)
    pipeline = build_pipeline(models[best_model_name](1))
    pipeline.fit(train_df, train_df["label"])

    joblib.dump(pipeline, MODELS_DIR / "best_model.joblib")

    metadata = {
        "model_name": best_model_name,
        "training_experiment": best_experiment,
        "primary_metric": "smishing_f1",
        "mean_smishing_f1": round(float(best["mean_smishing_f1"]), 4),
        "mean_macro_f1": round(float(best["mean_macro_f1"]), 4),
        "labels": ["ham", "spam", "smishing"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "best_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
