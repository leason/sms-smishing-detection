# Model Training

## Source: `src/models/train.py`, `src/features/text_features.py`

## Feature Engineering

### TF-IDF Text Features

Configuration in `src/features/text_features.py`:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=2,                 # ignore terms in fewer than 2 docs
    max_df=0.95,              # ignore terms in >95% of docs
    sublinear_tf=True         # apply 1 + log(tf)
)
```

Applied to the `text_clean` column.

### Structured Indicator Features

Three binary features passed through as-is:
- `has_url` (0/1)
- `has_email` (0/1)
- `has_phone` (0/1)

### Pipeline Assembly

`build_pipeline(classifier)` in `text_features.py` creates a scikit-learn `Pipeline`:

```
Pipeline([
    ("features", ColumnTransformer([
        ("tfidf", TfidfVectorizer(...), "text_clean"),
        ("indicators", "passthrough", ["has_url", "has_email", "has_phone"]),
    ])),
    ("clf", classifier),
])
```

The entire pipeline is serializable via joblib, so the saved model includes the fitted vectorizer, vocabulary, and classifier weights.

## Classifiers

Defined in `MODELS` dict in `src/models/train.py`:

| Name | Class | Configuration |
|------|-------|---------------|
| MultinomialNB | `MultinomialNB()` | Default params |
| LogisticRegression | `LogisticRegression(max_iter=2000, class_weight="balanced")` | Balanced class weights, increased iterations |
| LinearSVC | `CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000))` | Balanced weights, wrapped for probability support |

LinearSVC is wrapped in `CalibratedClassifierCV` so the saved model supports `predict_proba()`, which the demo app uses for confidence scores.

## Experimental Design

### Seeds

30 random seeds (1–30). Each seed produces deterministic train/test splits and model initialization.

### Training Strategies

All experiments evaluate against the **Manual Holdout** (real-world SMS):

| Experiment ID | Training Data | Test Data | Purpose |
|---|---|---|---|
| `manual_only` | Manual train split | Manual holdout | Baseline: real data on real data |
| `synthetic_only` | Synthetic train split (size-matched) | Manual holdout | Can synthetic data detect real smishing? |
| `combined` | Manual train + full synthetic train | Manual holdout | Does adding synthetic data help? |

### Train/Test Splitting

`make_splits()` in `train.py`:

1. Stratified 80/20 split on manual dataset (preserves class proportions)
2. Stratified 80/20 split on synthetic dataset
3. Manual holdout is the test set for all three experiments within a seed

```python
train_test_split(df, test_size=0.20, stratify=df["label"], random_state=seed)
```

### Size-Matching Control

The synthetic dataset (~10k rows) is larger than the manual dataset (~6k rows). To ensure `manual_only` vs `synthetic_only` comparisons reflect data source differences (not training set size), the `synthetic_only` experiment downsamples:

1. Compute size of manual training split (e.g., 4,759 rows)
2. Stratified random sample from synthetic training split to match that size
3. Same seed used for reproducibility

The `combined` experiment uses the **full** synthetic training split concatenated with the manual training split.

### Typical Split Sizes

| Experiment | n_train | n_test |
|------------|--------:|-------:|
| manual_only | ~4,759 | ~1,190 |
| synthetic_only | ~4,759 (matched) | ~1,190 |
| combined | ~11,151 | ~1,190 |

## Training Loop

`run_experiments()` iterates: seeds x models x experiments = 30 x 3 x 3 = **270 runs**.

For each run:
1. Create splits for the seed
2. Get train/test data for the experiment
3. Build pipeline with the classifier
4. `pipeline.fit(train_df, train_df["label"])`
5. `pipeline.predict(test_df)`
6. Extract metrics, confusion matrix, classification report
7. Append to results lists

After all runs: save metrics, generate summary tables, select and save best model.

## Best Model Selection

`_select_and_save_best_model()`:

1. Group metrics by `(model_name, experiment_id)`
2. Compute mean `smishing_f1` and mean `macro_f1` across 30 seeds
3. Sort by smishing_f1 descending, macro_f1 as tiebreaker
4. Winner is retrained on seed=1 training split and serialized

### Current Best Model

- **LinearSVC + combined** training strategy
- Mean smishing F1: 0.939
- Mean macro F1: 0.949
- Saved to: `outputs/models/best_model.joblib` (4.9 MB)
- Metadata: `outputs/models/best_model_metadata.json`

### Model Class Order

The fitted model's `.classes_` attribute is `['ham', 'smishing', 'spam']` (alphabetical). This is important for interpreting `predict_proba()` output — the app uses `model.classes_` to align probabilities with labels.

## Results Summary

| Model | manual_only | synthetic_only | combined |
|-------|-------------|----------------|----------|
| LinearSVC | 0.870 (0.023) | 0.930 (0.016) | 0.939 (0.015) |
| LogisticRegression | 0.868 (0.023) | 0.900 (0.016) | 0.925 (0.017) |
| MultinomialNB | 0.857 (0.019) | 0.887 (0.016) | 0.906 (0.020) |

*Mean smishing F1 (std dev) across 30 seeds.*
