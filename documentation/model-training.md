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
| LogisticRegression | `LogisticRegression(max_iter=2000, class_weight="balanced")` | Balanced class weights |
| LinearSVC | `CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000))` | Balanced weights, wrapped for probability support |

LinearSVC is wrapped in `CalibratedClassifierCV` so the saved model supports `predict_proba()`, which the demo app uses for confidence scores.

## Experimental Design

### Training strategies

Three strategies, all evaluated against the manual real-world holdout:

| Strategy | Training data |
|----------|---------------|
| `manual_only` | Manual train split |
| `synthetic_only` | Synthetic train split (full natural volume, no downsampling) |
| `combined` | Manual train + synthetic train (concatenated) |

The synthetic_only strategy uses the **full** natural-volume synthetic training split. No size-matching is applied — volume is a core advantage of LLM-generated data and matching away that advantage would conceal what the comparison is measuring.

### Seeds

30 random seeds (1–30). Each seed produces deterministic train/test splits and model initialization.

### Train/Test Splitting

`make_splits()` in `train.py`:

1. Stratified 80/20 split on manual dataset (preserves class proportions)
2. Stratified 80/20 split on synthetic dataset
3. Manual holdout is the test set for all three strategies within a seed

```python
train_test_split(df, test_size=0.20, stratify=df["label"], random_state=seed)
```

### Typical split sizes (per seed)

| Strategy | n_train | of which ham | spam | smishing |
|----------|--------:|-------------:|-----:|---------:|
| manual_only | 4,759 | 3,867 | 390 | 502 |
| synthetic_only | 6,392 | 2,714 | 1,555 | 2,123 |
| combined | 11,151 | 6,581 | 1,945 | 2,625 |

Manual holdout (test): ~1,190 rows across all strategies. Full per-seed composition is in `outputs/metrics/train_composition.csv`.

## Training Loop

`run_experiments()` iterates: seeds × models × strategies = 30 × 3 × 3 = **270 runs**.

For each run:
1. Create splits for the seed
2. Get train/test data for the strategy
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

### Current best model

- **LinearSVC + combined** training strategy
- Mean smishing F1: 0.939
- Mean macro F1: 0.949
- Saved to: `outputs/models/best_model.joblib`
- Metadata: `outputs/models/best_model_metadata.json` (includes `duplicate_mode`, `training_experiment`, headline metrics, trained_at)

This is the model deployed by the FastAPI demo app.

### Model class order

The fitted model's `.classes_` attribute is `['ham', 'smishing', 'spam']` (alphabetical). This is important for interpreting `predict_proba()` output — the app uses `model.classes_` to align probabilities with labels.

## Results Summary

Mean smishing F1 (std) across 30 seeds:

| Model | manual_only | synthetic_only | combined |
|-------|------------:|---------------:|---------:|
| Multinomial NB | 0.857 (0.019) | 0.900 (0.017) | 0.906 (0.020) |
| Logistic Regression | 0.868 (0.023) | 0.905 (0.016) | 0.925 (0.017) |
| LinearSVC | 0.870 (0.023) | 0.933 (0.015) | **0.939** (0.015) |

Mean macro F1 follows the same pattern (LinearSVC combined: 0.949). Full tables in `outputs/tables/model_performance_summary.csv` and `outputs/tables/experiment_performance_summary.csv`.

## Notes on framing

This comparison embraces the natural training-data volume produced by each pipeline rather than artificially controlling for size. The cross-dataset text overlap (specifically the 100% ham copy from manual into synthetic) is reported as a structural property of the synthetic corpus in the data-pipeline documentation, and the results are interpreted accordingly: the headline numbers describe practical performance on a holdout of the manual corpus, not generalization to unseen-content distributions. See the README's "Scope of the claim" bullet for details.
