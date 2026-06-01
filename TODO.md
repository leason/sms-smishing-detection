# Project Tracker: SMS Smishing Detection

> Working document for tracking tasks, status, and open questions.
> Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Workstream 1: Jupyter Notebook / ML Pipeline

### 1A. Project Scaffolding
- [x] Create repository directory structure (src/, data/, outputs/, notebooks/, r/)
- [x] Create `requirements.txt` with Python dependencies
- [x] Place raw datasets in `data/raw/`

### 1B. Data Preprocessing (`src/data/preprocessing.py`)
- [x] Load raw CSVs and add `dataset_source` column
- [x] Normalize column names (LABEL->label, TEXT->text, etc.)
- [x] Normalize labels (Smishing->smishing, Spam->spam)
- [x] Normalize indicator columns to binary integers
- [x] Text cleaning -> `text_clean` field
- [x] Implement duplicate-handling modes (keep_duplicates, drop_exact_duplicates, overlap_aware)
- [x] Leakage detection: identify exact overlap between datasets
- [x] Save processed files (manual_clean.csv, synthetic_clean.csv, combined_clean.csv)
- [x] Validate: no unexpected labels, no missing TEXT/LABEL

### 1C. Dataset Audit (`src/data/audit.py`)
- [x] Generate `outputs/tables/dataset_audit.csv` (row counts, class balance, duplicates, vocab size, etc.)
- [x] Generate `outputs/tables/cross_dataset_overlap.csv`
- [x] Generate `outputs/tables/top_duplicate_messages.csv`
- [x] Generate `outputs/tables/leakage_audit.csv`
- [x] Generate audit figures:
  - [x] class_balance_comparison.png
  - [x] duplicate_rate_comparison.png
  - [x] dataset_overlap_summary.png
  - [x] message_length_distribution.png

### 1D. Feature Engineering (`src/features/text_features.py`)
- [x] TF-IDF vectorizer config (bigrams, sublinear_tf, etc.)
- [x] ColumnTransformer combining TF-IDF + indicator features (has_url, has_email, has_phone)
- [x] Wrap in scikit-learn pipeline

### 1E. Model Training (`src/models/train.py`)
- [x] Multinomial Naive Bayes pipeline
- [x] Logistic Regression pipeline
- [x] Linear SVM pipeline (wrapped in CalibratedClassifierCV)
- [ ] Optional: Random Forest pipeline
- [x] Training loop: 30 seeds x 3 experiments x 3 models (270 runs complete)
- [x] Size-matching control: downsample synthetic training split for `synthetic_only`
- [x] Stratified 80/20 train/test splits per seed

### 1F. Evaluation (`src/models/evaluate.py`, `src/models/figures.py`)
- [x] Per-run metrics: accuracy, macro P/R/F1, per-class P/R/F1, n_train, n_test
- [x] Save `outputs/metrics/all_model_results.csv`
- [x] Save `outputs/metrics/confusion_matrices.csv`
- [x] Save `outputs/metrics/classification_reports.jsonl`
- [x] Generate summary tables:
  - [x] outputs/tables/model_performance_summary.csv
  - [x] outputs/tables/experiment_performance_summary.csv
- [x] Generate evaluation figures:
  - [x] mean_smishing_f1_by_training_strategy.png
  - [x] mean_macro_f1_by_training_strategy.png
  - [x] confusion_matrix_best_model_manual_holdout.png
  - [x] smishing_f1_boxplot_by_training_strategy.png

### 1G. Best Model Selection
- [x] Rank by mean smishing F1 on manual holdout, macro F1 tiebreaker
- [x] Save `outputs/models/best_model.joblib`
- [x] Save `outputs/models/best_model_metadata.json`

### 1H. Notebook Assembly
- [x] Create `notebooks/01_model_training_and_evaluation.ipynb`
- [x] Sections: overview, loading, audit, duplicates, overlap, class dist, features, training, eval, stats summary, model selection, export
- [x] Notebook calls `src/` modules (not all logic inline)
- [x] Verified: notebook executes end-to-end via nbconvert

---

## Workstream 2: R Statistical Analysis

### 2A. Script Setup (`r/statistical_analysis.R`)
- [x] Read `outputs/metrics/all_model_results.csv`
- [x] Filter/reshape data for paired comparisons

### 2B. Hypothesis Tests
- [x] Paired t-test and/or Wilcoxon signed-rank for each comparison:
  - [x] manual_only vs synthetic_only
  - [x] manual_only vs combined
  - [x] synthetic_only vs combined
- [x] Two-tailed testing
- [x] Benjamini-Hochberg FDR correction
- [x] Per-model AND pooled (ALL) comparisons
- [x] Shapiro-Wilk normality check to select t-test vs Wilcoxon

### 2C. Effect Sizes
- [x] Cohen's d for paired differences
- [x] Wilcoxon r effect size

### 2D. Outputs
- [x] Save `outputs/tables/statistical_tests.csv`
- [x] Save `outputs/tables/effect_sizes.csv`
- [x] Save `outputs/figures/statistical_comparison_boxplots.png`

---

## Workstream 3: FastAPI Demo App

### 3A. Application (`src/app/main.py`)
- [x] Load best_model.joblib on startup
- [x] GET `/` — HTML form with textarea + sample messages
- [x] POST `/predict` — form submission, returns HTML result page
- [x] POST `/api/predict` — JSON API endpoint
- [x] GET `/health` — health check
- [x] Deterministic explanation engine (URL/email/phone detection, keyword matching)
- [x] Confidence/probability handling (CalibratedClassifierCV, uses model.classes_ for label alignment)

### 3B. Frontend (`src/app/templates/index.html`)
- [x] SMS input form
- [x] Result display (prediction, confidence, probability bars, indicators)
- [x] 6 sample messages (2 ham, 1 spam, 3 smishing variants)

### 3C. Docker
- [x] Dockerfile (python:3.11-slim, uvicorn entrypoint)
- [x] docker-compose.yml (port 8000, model volume mount)
- [x] Verified: `docker compose up --build` works end-to-end

### 3D. README
- [x] Project title and research question
- [x] Dataset descriptions (including provenance)
- [x] Setup instructions (venv, pip install)
- [x] How to run: preprocessing, training, R analysis, demo app
- [x] Project structure
- [x] Output summary
- [x] Limitations

---

## Open Questions / Decisions
- (none yet)

---

## Key Results (30-seed averages)
| Model | manual_only | synthetic_only | combined |
|-------|-------------|----------------|----------|
| LinearSVC | 0.870 | 0.930 | 0.939 |
| LogisticRegression | 0.868 | 0.900 | 0.925 |
| MultinomialNB | 0.857 | 0.887 | 0.906 |

Best model: **LinearSVC combined** (mean smishing F1 = 0.939)

## Notes
- Primary analysis uses `drop_exact_duplicates` mode
- All experiments evaluate against Manual Holdout (real-world SMS)
- `synthetic_only` uses size-matched training data (4759 rows, matching manual)
- `combined` uses full synthetic + manual training data (11151 rows)
- 70% of manual texts appear in synthetic dataset (high overlap)
- Synthetic-only consistently outperforms manual-only across all models
