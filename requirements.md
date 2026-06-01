# Requirements: Evaluating LLM-Generated Synthetic Datasets for SMS Smishing Detection

## 1. Project Overview

This project evaluates the effectiveness of an LLM-generated synthetic SMS dataset for training machine learning models that detect smishing messages in real-world SMS data. The synthetic dataset was generated using the manually curated dataset as source material, which introduces overlap and leakage concerns that the implementation must address.

The implementation must produce three major outputs:

1. A Python/Jupyter machine learning workflow that trains and evaluates multiclass SMS classifiers across training-strategy permutations.
2. An R statistical analysis workflow that performs hypothesis testing on model performance results.
3. A small locally deployed FastAPI web application, packaged with Docker, that allows a user to paste an SMS message and receive a model prediction.

An iOS SMS filtering extension may be explored as a stretch goal, but it is not part of the required implementation path.

---

## 2. Research Objectives

### 2.1 Primary Research Question

Can an LLM-generated synthetic SMS dataset be used to effectively train machine learning models for smishing detection on real-world SMS messages?

### 2.2 Secondary Research Questions

1. Do models trained on synthetic SMS data achieve comparable performance to models trained on manually curated SMS data when evaluated on real SMS messages?
2. Does combining synthetic and manually curated SMS data change performance?
3. What characteristics distinguish the synthetic dataset from the manually curated dataset?

### 2.3 Null Hypothesis

There is no statistically significant difference in smishing detection performance between models trained on manually curated SMS data and models trained on LLM-generated SMS data when evaluated on real SMS messages.

### 2.4 Alternative Hypothesis

There is a statistically significant difference in smishing detection performance between models trained on manually curated SMS data and models trained on LLM-generated SMS data when evaluated on real SMS messages.

### 2.5 Primary Outcome Metric

The primary metric is:

- F1-score for the `smishing` class

### 2.6 Secondary Metrics

The secondary metrics are:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision, recall, and F1-score for:
  - `ham`
  - `spam`
  - `smishing`
- Confusion matrix values

---

## 3. Datasets

### 3.1 Input Files

The project assumes two source CSV files in the `data/raw/` directory:

```text
data/raw/Dataset_5971.csv
data/raw/Dataset_10191.csv
```

### 3.2 Dataset Source Definitions

| File | Dataset Source | Description |
|---|---|---|
| `Dataset_5971.csv` | `manual` | Manually curated SMS dataset (real-world messages) |
| `Dataset_10191.csv` | `synthetic` | LLM-generated synthetic SMS dataset |

### 3.3 Dataset Provenance

The synthetic dataset was generated using the manually curated dataset as source material. This means:

1. There may be exact text overlap between the two datasets.
2. The synthetic dataset's distribution and content patterns are influenced by the manual dataset.
3. Leakage prevention is critical when designing train/test splits.

### 3.4 Observed Column Schema

Both CSV files contain the same columns:

| Column | Type | Meaning |
|---|---|---|
| `LABEL` | string | SMS class label |
| `TEXT` | string | SMS message body |
| `URL` | string | Whether message contains a URL |
| `EMAIL` | string | Whether message contains an email address |
| `PHONE` | string | Whether message contains a phone number |

### 3.5 Observed Label Distributions

#### Dataset_5971.csv

Raw labels observed:

| Raw Label | Count |
|---|---:|
| `ham` | 4,844 |
| `Smishing` | 616 |
| `spam` | 466 |
| `Spam` | 23 |
| `smishing` | 22 |

Normalized class counts should become:

| Normalized Label | Count |
|---|---:|
| `ham` | 4,844 |
| `smishing` | 638 |
| `spam` | 489 |

#### Dataset_10191.csv

Raw labels observed:

| Raw Label | Count |
|---|---:|
| `ham` | 3,397 |
| `smishing` | 3,397 |
| `spam` | 3,397 |

### 3.6 Data Quality Observations

The implementation must account for the following observed data issues:

1. `Dataset_5971.csv` uses inconsistent capitalization for labels:
   - `Smishing` and `smishing`
   - `Spam` and `spam`

2. `Dataset_5971.csv` contains 22 duplicate `TEXT` values.

3. `Dataset_10191.csv` contains 2,200 duplicate `TEXT` values.

4. The indicator columns use inconsistent capitalization:
   - `yes` / `No` in `Dataset_5971.csv`
   - `Yes` / `No` in `Dataset_10191.csv`

5. No missing values were observed in either dataset, but the code should still validate for missing `TEXT` or `LABEL` fields.

### 3.7 Dataset Provenance Audit

Before modeling, the implementation must perform a dataset audit and generate:

```text
outputs/tables/dataset_audit.csv
outputs/tables/cross_dataset_overlap.csv
outputs/tables/top_duplicate_messages.csv
outputs/tables/leakage_audit.csv
```

Required audit metrics:

- Row count
- Class balance
- Exact duplicate count
- Duplicate percentage
- Conflicting label count
- Average message length
- Vocabulary size
- Cross-dataset overlap count

Required audit figures:

```text
outputs/figures/class_balance_comparison.png
outputs/figures/duplicate_rate_comparison.png
outputs/figures/dataset_overlap_summary.png
outputs/figures/message_length_distribution.png
```

---

## 4. Required Repository Structure

The generated source code should use the following repository structure:

```text
.
├── README.md
├── requirements.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/
│   │   ├── Dataset_5971.csv
│   │   └── Dataset_10191.csv
│   ├── processed/
│   │   ├── manual_clean.csv
│   │   ├── synthetic_clean.csv
│   │   └── combined_clean.csv
│   └── splits/
├── notebooks/
│   └── 01_model_training_and_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── text_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── templates/
│   │       └── index.html
│   └── utils/
│       ├── __init__.py
│       └── paths.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── tables/
└── r/
    └── statistical_analysis.R
```

---

## 5. Python Environment Requirements

The project should use Python 3.11 or later.

### 5.1 Required Python Packages

Include the following in `requirements.txt`:

```text
fastapi
uvicorn[standard]
jinja2
python-multipart
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter
ipykernel
scipy
```

Optional packages:

```text
statsmodels
pytest
black
ruff
```

---

## 6. Data Preprocessing Requirements

### 6.1 Load Raw Data

Create reusable preprocessing functions that:

1. Load `Dataset_5971.csv`.
2. Load `Dataset_10191.csv`.
3. Add a `dataset_source` column:
   - `manual` for `Dataset_5971.csv`
   - `synthetic` for `Dataset_10191.csv`

### 6.2 Normalize Column Names

Convert raw columns into standardized lowercase names:

| Raw Column | Standardized Column |
|---|---|
| `LABEL` | `label` |
| `TEXT` | `text` |
| `URL` | `has_url` |
| `EMAIL` | `has_email` |
| `PHONE` | `has_phone` |

### 6.3 Normalize Labels

Normalize labels as follows:

| Raw Label | Normalized Label |
|---|---|
| `ham` | `ham` |
| `spam` | `spam` |
| `Spam` | `spam` |
| `smishing` | `smishing` |
| `Smishing` | `smishing` |

Any label outside `ham`, `spam`, or `smishing` should raise a validation error.

### 6.4 Normalize Indicator Columns

Normalize `has_url`, `has_email`, and `has_phone` to binary integers:

| Raw Value | Normalized Value |
|---|---:|
| `Yes` | 1 |
| `yes` | 1 |
| `Y` | 1 |
| `true` | 1 |
| `1` | 1 |
| `No` | 0 |
| `no` | 0 |
| `N` | 0 |
| `false` | 0 |
| `0` | 0 |

Any unrecognized value should raise a validation warning and be treated as missing.

### 6.5 Text Cleaning

Create a cleaned text field named `text_clean`.

Required transformations:

1. Convert to string.
2. Strip leading/trailing whitespace.
3. Replace repeated internal whitespace with a single space.
4. Preserve casing for TF-IDF vectorization unless the vectorizer handles lowercase conversion.
5. Do not remove URLs, phone numbers, or email addresses by default because these may be predictive security indicators.

### 6.6 Duplicate and Overlap Handling

The pipeline should support three duplicate-handling modes:

| Mode | Description |
|---|---|
| `keep_duplicates` | Keep all rows. Used for sensitivity analysis. |
| `drop_exact_duplicates` | Drop duplicate messages based on exact raw `text`, retaining the first occurrence. This is the primary analysis mode. We use raw text because differences in whitespace may be material. |
| `overlap_aware` | Remove messages from the synthetic dataset that also appear in the manual dataset. Prevents leakage from shared content between the datasets. |

The primary analysis should use `drop_exact_duplicates` to reduce leakage risk, especially because the synthetic dataset contains a large number of duplicate texts.

A sensitivity analysis may also be run with duplicates retained.

### 6.7 Leakage Prevention

Because the synthetic dataset was generated from the manual dataset:

1. Identify exact text overlap between the two datasets.
2. Report overlap statistics in the dataset audit.
3. Ensure holdout messages from the manual dataset never appear in the synthetic training data.
4. Generate a leakage audit report at `outputs/tables/leakage_audit.csv`.

### 6.8 Save Processed Files

The preprocessing step must save:

```text
data/processed/manual_clean.csv
data/processed/synthetic_clean.csv
data/processed/combined_clean.csv
```

Each processed file must contain:

```text
dataset_source
label
text
text_clean
has_url
has_email
has_phone
```

---

## 7. Feature Engineering Requirements

### 7.1 Text Features

Use TF-IDF vectorization as the primary feature representation.

Default TF-IDF configuration:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

### 7.2 Structured Indicator Features

Include the following binary features as additional model inputs:

- `has_url`
- `has_email`
- `has_phone`

These should be combined with TF-IDF features using a scikit-learn `ColumnTransformer` or equivalent pipeline.

### 7.3 Pipeline Requirement

All models must be implemented as scikit-learn pipelines so that preprocessing, vectorization, feature combination, and classification are reproducible and serializable.

---

## 8. Models to Train

Train and evaluate the following multiclass classifiers:

### 8.1 Multinomial Naive Bayes

Use:

```python
MultinomialNB()
```

### 8.2 Logistic Regression

Use:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    multi_class="auto"
)
```

### 8.3 Linear Support Vector Machine

Use:

```python
LinearSVC(
    class_weight="balanced"
)
```

### 8.4 Optional Model

Random Forest may be included as an optional comparison model, but it is not required.

If included, use:

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=seed,
    n_jobs=-1
)
```

---

## 9. Experimental Design

### 9.1 Random Seeds

Run repeated experiments across 30 random seeds:

```text
1 through 30
```

Each seed should produce one complete set of training and evaluation results for each model and each training strategy.

### 9.2 Training Strategies

All experiments are evaluated against the Manual Holdout set because it represents real-world SMS messages.

For each model and seed, run the following experiments:

| Experiment ID | Training Data | Test Data | Purpose |
|---|---|---|---|
| `manual_only` | Manual train split | Manual holdout | Baseline: real data trained and tested on real data |
| `synthetic_only` | Synthetic train split (size-matched) | Manual holdout | Can synthetic data alone detect real smishing? |
| `combined` | Manual train + Synthetic train | Manual holdout | Does adding synthetic data improve performance? |

### 9.3 Training Data Size Control

The synthetic dataset (~10,191 rows) is substantially larger than the manual dataset (~5,971 rows). To ensure that comparisons between `manual_only` and `synthetic_only` reflect differences in data source rather than training set size, the `synthetic_only` experiment must use a stratified random sample of the synthetic training split that matches the size of the manual training split.

For each seed:

1. After creating the synthetic train/test split, compute the size of the manual training split.
2. Draw a stratified random sample (by `label`) from the synthetic training split to match that size.
3. Use the same seed for reproducibility.

The `combined` experiment should use the full synthetic training split (not size-matched) concatenated with the manual training split, since the research question for that experiment is whether adding more synthetic data helps.

### 9.4 Split Strategy

For each seed:

1. Create a stratified train/test split for the manual dataset.
2. Create a stratified train/test split for the synthetic dataset.
3. Use an 80/20 split.
4. Preserve the same manual holdout split for all experiments within a seed.
5. Ensure that duplicate text messages are not split across training and testing when duplicate removal is disabled.

Preferred implementation:

```python
train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=seed
)
```

### 9.5 Combined Training Data

For combined experiments:

1. Use the manual training split.
2. Use the full (not size-matched) synthetic training split.
3. Concatenate them into one training set.
4. Do not include the manual holdout rows in the combined training set.

### 9.6 Model Selection for Demo App

The demo app should use the trained model with the best average smishing-class F1-score on the manual holdout set.

Preferred selection criterion:

1. First rank by mean smishing F1 across seeds.
2. Use macro-F1 as a tie-breaker.
3. Save the selected model to:

```text
outputs/models/best_model.joblib
```

Also save a metadata file:

```text
outputs/models/best_model_metadata.json
```

Metadata should include:

```json
{
  "model_name": "LinearSVC",
  "training_experiment": "combined",
  "primary_metric": "smishing_f1",
  "mean_smishing_f1": 0.0,
  "mean_macro_f1": 0.0,
  "labels": ["ham", "spam", "smishing"],
  "trained_at": "ISO-8601 timestamp"
}
```

---

## 10. Evaluation Output Requirements

### 10.1 Metrics CSV

The Python evaluation pipeline must save a long-format metrics CSV:

```text
outputs/metrics/all_model_results.csv
```

Each row should represent one unique combination of:

- seed
- model_name
- experiment_id
- duplicate_mode

Required columns:

```text
seed
model_name
experiment_id
duplicate_mode
n_train
n_test
accuracy
macro_precision
macro_recall
macro_f1
ham_precision
ham_recall
ham_f1
spam_precision
spam_recall
spam_f1
smishing_precision
smishing_recall
smishing_f1
```

`n_train` and `n_test` record the exact number of rows used for training and testing in that run. These columns allow verification of the size-matching control in `synthetic_only` experiments and support analysis of whether training set size influences results.

### 10.2 Confusion Matrix Output

Save confusion matrices in long format:

```text
outputs/metrics/confusion_matrices.csv
```

Required columns:

```text
seed
model_name
experiment_id
true_label
predicted_label
count
```

### 10.3 Classification Reports

Save one detailed classification report per model/experiment/seed as JSON or save all reports in one combined JSONL file:

```text
outputs/metrics/classification_reports.jsonl
```

### 10.4 Summary Tables

Generate summary tables:

```text
outputs/tables/model_performance_summary.csv
outputs/tables/experiment_performance_summary.csv
```

Summary tables should include:

- mean
- standard deviation
- median
- interquartile range
- minimum
- maximum

for all primary and secondary metrics.

### 10.5 Figures

Generate and save the following figures:

```text
outputs/figures/class_balance_comparison.png
outputs/figures/duplicate_rate_comparison.png
outputs/figures/dataset_overlap_summary.png
outputs/figures/message_length_distribution.png
outputs/figures/mean_smishing_f1_by_training_strategy.png
outputs/figures/mean_macro_f1_by_training_strategy.png
outputs/figures/confusion_matrix_best_model_manual_holdout.png
```

Optional figures:

```text
outputs/figures/smishing_f1_boxplot_by_training_strategy.png
```

---

## 11. Statistical Analysis Requirements

### 11.1 Input File

The R analysis should read:

```text
outputs/metrics/all_model_results.csv
```

### 11.2 Statistical Questions

The R script should answer the following questions:

1. Does smishing F1-score differ between `manual_only` and `synthetic_only`?
2. Does `combined` outperform `manual_only`?
3. Does `combined` outperform `synthetic_only`?
4. Are observed differences practically meaningful?

### 11.3 Primary Hypothesis Test

The primary hypothesis test should compare smishing-class F1-score across training strategies.

Use a two-tailed paired test where comparisons are paired by:

- seed
- model_name

Preferred tests:

1. Paired t-test if normality of paired differences is reasonable.
2. Wilcoxon signed-rank test if normality is questionable.

### 11.4 Required Comparisons

At minimum, perform the following paired comparisons:

| Comparison | Description |
|---|---|
| `manual_only` vs `synthetic_only` | Does training data source matter? |
| `manual_only` vs `combined` | Does adding synthetic data help? |
| `synthetic_only` vs `combined` | Does adding real data help? |

### 11.5 Effect Sizes

Report effect sizes for each comparison.

Preferred effect sizes:

- Cohen's d for paired differences
- Wilcoxon effect size if Wilcoxon is used

### 11.6 Multiple Comparisons

Apply a multiple-comparison correction across the required tests.

Preferred correction:

```text
Benjamini-Hochberg FDR correction
```

### 11.7 R Output Files

The R script must save:

```text
outputs/tables/statistical_tests.csv
outputs/tables/effect_sizes.csv
outputs/figures/statistical_comparison_boxplots.png
```

### 11.8 Statistical Test Output Columns

`statistical_tests.csv` should include:

```text
comparison
metric
model_name
test_type
n_pairs
mean_difference
median_difference
statistic
p_value
p_value_adjusted
significant_at_0_05
```

---

## 12. Jupyter Notebook Requirements

Create:

```text
notebooks/01_model_training_and_evaluation.ipynb
```

The notebook should be readable as a project artifact and should include:

1. Project title and objective
2. Dataset loading
3. Dataset audit and provenance analysis
4. Duplicate analysis
5. Cross-dataset overlap analysis
6. Class distribution visualizations
7. Train/test split explanation
8. Feature engineering
9. Model training
10. Model evaluation
11. Statistical summary
12. Best model selection
13. Export of metrics, figures, and serialized model

The notebook should call reusable functions from `src/` rather than containing all logic inline.

---

## 13. FastAPI Demo App Requirements

### 13.1 Purpose

The FastAPI app provides a local demo interface for classifying pasted SMS messages using the best trained model.

The app does not need to send or receive real SMS messages.

### 13.2 Deployment Target

The app must run locally in Docker.

Required local URL:

```text
http://localhost:8000
```

### 13.3 App Files

Required files:

```text
src/app/main.py
src/app/templates/index.html
Dockerfile
docker-compose.yml
```

### 13.4 Required Endpoints

#### GET `/`

Returns a simple HTML form with:

- Textarea for SMS message
- Submit button
- Optional sample messages for demonstration

#### POST `/predict`

Accepts form data:

```text
message=<sms text>
```

Returns an HTML page showing:

- Original message
- Predicted class
- Confidence score if available
- Class probabilities if available
- Simple deterministic explanation based on:
  - whether the message contains a URL
  - whether the message contains a phone number
  - whether the message contains an email address
  - suspicious keywords
  - model prediction

#### GET `/health`

Returns:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### POST `/api/predict`

Accepts JSON:

```json
{
  "message": "Your account has been locked. Verify now at http://example.com"
}
```

Returns JSON:

```json
{
  "message": "Your account has been locked. Verify now at http://example.com",
  "prediction": "smishing",
  "confidence": 0.92,
  "probabilities": {
    "ham": 0.01,
    "spam": 0.07,
    "smishing": 0.92
  },
  "indicators": {
    "has_url": true,
    "has_email": false,
    "has_phone": false,
    "suspicious_keywords": ["account", "locked", "verify"]
  }
}
```

If the selected model does not support calibrated probabilities, the app should either:

1. return decision scores, or
2. wrap the model in a calibration step during training, or
3. return `confidence: null` and explain that the model does not provide probabilities.

Preferred approach:

- Use calibrated probabilities for `LinearSVC` through `CalibratedClassifierCV`, or select Logistic Regression if probability output is important for the demo.

### 13.5 Demo Explanation Rules

Do not use an LLM for explanation in the required app.

Instead, use deterministic indicators.

Suspicious keyword list should include at least:

```text
urgent
verify
account
locked
suspended
click
login
password
bank
refund
delivery
package
prize
winner
claim
payment
tax
security
limited time
```

The explanation should be clearly labeled as heuristic, not as proof that the message is malicious.

### 13.6 Example Demo Messages

Include at least six sample messages in the app:

1. Benign casual message
2. Benign appointment/reminder message
3. Spam promotional message
4. Smishing bank/account message
5. Smishing package delivery message
6. Smishing prize/refund message

---

## 14. Docker Requirements

### 14.1 Dockerfile

The Dockerfile should:

1. Use a slim Python base image.
2. Install Python dependencies.
3. Copy the project source.
4. Expose port 8000.
5. Start the FastAPI app with Uvicorn.

Expected command:

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### 14.2 docker-compose.yml

`docker-compose.yml` should define one service:

```yaml
services:
  smishing-demo:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs/models:/app/outputs/models
```

---

## 15. README Requirements

The README must include:

1. Project title
2. Research question
3. Dataset descriptions (including provenance relationship)
4. Setup instructions
5. How to run preprocessing
6. How to run model training
7. How to run statistical analysis
8. How to run the FastAPI demo
9. Project structure
10. Summary of outputs
11. Notes on limitations

### 15.1 Required Commands

The README should include commands similar to:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
jupyter notebook notebooks/01_model_training_and_evaluation.ipynb
```

```bash
Rscript r/statistical_analysis.R
```

```bash
docker compose up --build
```

---

## 16. Reproducibility Requirements

The implementation must be reproducible.

Required controls:

1. Fixed random seeds.
2. Saved train/test split identifiers or deterministic splitting.
3. Serialized final model.
4. Saved model metadata.
5. All metrics written to CSV.
6. All statistical test outputs written to CSV.
7. All figures written to image files.

---

## 17. Acceptance Criteria

The implementation is complete when all of the following are true:

### 17.1 Dataset Audit

- Dataset provenance audit completed.
- Leakage audit completed.
- Cross-dataset overlap reported.
- Audit tables and figures generated.

### 17.2 Data Processing

- Raw datasets load successfully.
- Labels normalize into exactly three classes:
  - `ham`
  - `spam`
  - `smishing`
- Indicator columns normalize to binary values.
- Processed CSV files are saved.

### 17.3 Model Training

- Multinomial Naive Bayes trains successfully.
- Logistic Regression trains successfully.
- Linear SVM trains successfully.
- Each model runs across all required training strategies.
- Experiments run across 30 random seeds.

### 17.4 Evaluation

- `outputs/metrics/all_model_results.csv` is generated.
- `outputs/metrics/confusion_matrices.csv` is generated.
- Summary tables are generated.
- Required figures are generated.
- Best model is saved as `outputs/models/best_model.joblib`.

### 17.5 Statistical Analysis

- R script reads Python metrics output.
- Required paired comparisons are performed.
- Two-tailed p-values are reported.
- Effect sizes are reported.
- Multiple-comparison correction is applied.
- Statistical result CSVs are saved.

### 17.6 Demo App

- Docker container builds successfully.
- FastAPI app starts locally.
- User can paste an SMS message into a web form.
- App returns a predicted class.
- App displays deterministic risk indicators.
- Health endpoint returns status successfully.

---

## 18. Out of Scope

The following are out of scope for the required implementation:

1. Real-time SMS ingestion.
2. Production iOS app deployment.
3. iMessage integration.
4. LLM-generated explanations.
5. Cloud deployment.
6. User authentication.
7. Persistent user accounts.
8. Database storage.
9. Automated SMS sending through Twilio.
10. Real-world operational security monitoring.

---

## 19. Stretch Goal: iOS SMS Filtering Extension

If time permits, a separate iOS proof of concept may be created.

### 19.1 Stretch Goal Objective

Create an iOS SMS/MMS filtering extension that classifies unknown incoming SMS messages using either:

1. a simplified local heuristic classifier, or
2. an exported model compatible with the iOS app.

### 19.2 Stretch Goal Requirements

The iOS extension should:

1. Receive unknown SMS/MMS messages through Apple's Message Filter Extension API.
2. Classify messages into benign or suspicious categories.
3. Write filtered message metadata to shared App Group storage if technically feasible.
4. Allow the companion app to display a local log of filtered messages.

### 19.3 Stretch Goal Constraints

The iOS extension is not required for the course deliverable.

The main project must remain complete and demoable without this stretch goal.

---

## 20. Implementation Guidance for Coding Assistants

When using an AI coding assistant to implement this project:

1. Build the preprocessing and dataset audit module first.
2. Validate both datasets and generate audit outputs before training.
3. Implement model pipelines using scikit-learn.
4. Save all outputs to deterministic file paths.
5. Avoid adding unnecessary infrastructure.
6. Keep the FastAPI app simple and local-only.
7. Do not add LLM dependencies unless explicitly requested later.
8. Do not make the iOS extension part of the required build.
9. Prefer clarity and reproducibility over optimization.
10. Treat the statistical analysis as a required deliverable, not an optional add-on.

---

## 21. Suggested Build Order

1. Create repository structure.
2. Add raw datasets to `data/raw/`.
3. Implement preprocessing functions.
4. Run dataset provenance audit.
5. Generate processed datasets.
6. Implement model training pipelines.
7. Run one seed end-to-end.
8. Expand to 30 seeds.
9. Save metrics and confusion matrices.
10. Build summary figures.
11. Write R statistical analysis script.
12. Select and serialize best model.
13. Build FastAPI app.
14. Add Docker support.
15. Write README.
16. Run full reproducibility check from a clean checkout.
