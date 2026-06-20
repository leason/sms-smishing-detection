# Project Tracker: SMS Smishing Detection

> Working document for tracking tasks, status, and open questions.
> Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Workstream 1: ML Pipeline (Python)

### 1A. Project Scaffolding
- [x] Repository directory structure (src/, data/, outputs/, notebooks/, scripts/, r/)
- [x] `requirements.txt` with Python dependencies
- [x] Raw datasets in `data/raw/`

### 1B. Data Preprocessing (`src/data/preprocessing.py`)
- [x] Load raw CSVs and tag with `dataset_source`
- [x] Normalize column names, labels, indicators
- [x] Text cleaning → `text_clean` field
- [x] Within-dataset exact-text deduplication
- [x] Save processed files (manual_clean.csv, synthetic_clean.csv, combined_clean.csv)
- [x] Validation: no unexpected labels, no missing TEXT/LABEL

### 1C. Dataset Audit (`src/data/audit.py`)
- [x] `outputs/tables/dataset_audit.csv`
- [x] `outputs/tables/cross_dataset_overlap.csv`
- [x] `outputs/tables/top_duplicate_messages.csv`
- [x] `outputs/tables/leakage_audit.csv`
- [x] Audit figures (class balance, duplicate rate, overlap, message length)

### 1D. Feature Engineering (`src/features/text_features.py`)
- [x] TF-IDF (1,2)-grams, sublinear_tf, min_df=2, max_df=0.95
- [x] ColumnTransformer combining TF-IDF + indicator features
- [x] Wrap in scikit-learn pipeline

### 1E. Model Training (`src/models/train.py`)
- [x] Multinomial Naive Bayes pipeline
- [x] Logistic Regression pipeline (balanced)
- [x] LinearSVC pipeline (CalibratedClassifierCV, balanced)
- [x] Training loop: 30 seeds × 3 models × 3 strategies = 270 runs
- [x] Stratified 80/20 train/test splits per seed
- [x] No size matching — full natural volume per strategy

### 1F. Evaluation (`src/models/evaluate.py`, `src/models/figures.py`)
- [x] Per-run metrics: accuracy, macro P/R/F1, per-class P/R/F1, n_train, n_test
- [x] `outputs/metrics/all_model_results.csv`
- [x] `outputs/metrics/confusion_matrices.csv`
- [x] `outputs/metrics/classification_reports.jsonl`
- [x] `outputs/metrics/train_composition.csv`
- [x] Summary tables (per-model, per-experiment)
- [x] Evaluation figures (mean F1 by strategy, confusion matrix, boxplots) with `_data.csv` sidecars

### 1G. Best Model Selection
- [x] Rank by mean smishing F1, macro F1 tiebreaker
- [x] `outputs/models/best_model.joblib`
- [x] `outputs/models/best_model_metadata.json`

### 1H. Driver scripts
- [x] `scripts/run_experiments.py` — end-to-end runner
- [x] `notebooks/01_model_training_and_evaluation.ipynb` — Jupyter equivalent

---

## Workstream 2: R Statistical Analysis

### 2A. Script Setup (`r/statistical_analysis.R`)
- [x] Read `outputs/metrics/all_model_results.csv`
- [x] Within-strategy pairwise comparisons (3 comparisons × 4 groupings = 12 tests)

### 2B. Hypothesis Tests
- [x] Paired t-test / Wilcoxon signed-rank with Shapiro-Wilk gating
- [x] Benjamini-Hochberg FDR correction
- [x] Per-model AND pooled (ALL) comparisons

### 2C. Effect Sizes
- [x] Cohen's d (paired)
- [x] Wilcoxon r

### 2D. Outputs
- [x] `outputs/tables/statistical_tests.csv`
- [x] `outputs/tables/effect_sizes.csv`
- [x] `outputs/figures/statistical_comparison_boxplots.png` + `_data.csv` sidecar

---

## Workstream 3: FastAPI Demo App

### 3A. Application (`src/app/main.py`)
- [x] Load best_model.joblib on startup
- [x] GET `/` — HTML form with textarea + sample messages
- [x] POST `/predict` — form submission, returns HTML result page
- [x] POST `/api/predict` — JSON API endpoint
- [x] GET `/health` — health check
- [x] GET `/api/model` — deployed-model metadata
- [x] Deterministic explanation engine (URL/email/phone detection, keyword matching)
- [x] Confidence/probability handling (uses `model.classes_`)

### 3B. Frontend (`src/app/templates/index.html`)
- [x] SMS input form
- [x] Result display (prediction, confidence, probability bars, indicators)
- [x] 6 sample messages
- [x] Deployed-model info footer

### 3C. Docker
- [x] Dockerfile (python:3.11-slim, uvicorn entrypoint)
- [x] docker-compose.yml (port 8000, model volume mount read-only)
- [x] Verified: `docker compose up --build` works end-to-end

---

## Workstream 4: Documentation
- [x] README.md
- [x] documentation/README.md (index)
- [x] documentation/architecture.md
- [x] documentation/data-pipeline.md
- [x] documentation/model-training.md
- [x] documentation/statistical-analysis.md
- [x] documentation/demo-app.md
- [x] documentation/outputs-reference.md
- [x] documentation/ios-app-proposal.md
- [x] project-report-draft.md
- [x] requirements.md (project specification)

---

## Workstream 5: iOS App (Stretch Goal)

### 5A. Research Spike
- [x] Test coremltools conversion — BLOCKED (requires sklearn ≤1.5.1)
- [x] Evaluate simplified models (5k, 10k features) — 5k loses <1% F1
- [x] Choose approach: Swift-native TF-IDF + exported weights (Option D)
- [x] Export model weights as JSON (489 KB)
- [x] Generate Python reference predictions for parity testing

### 5B. Swift Classifier (`ios/SMSShield/Shared/`)
- [x] SMSClassifier.swift — TF-IDF + LogReg inference in Swift
- [x] FilteredMessage.swift — data model for logged messages
- [x] MessageStore.swift — App Group JSON persistence
- [x] IndicatorDetector.swift — URL/email/phone/keyword detection

### 5C. Message Filter Extension
- [x] MessageFilterExtension.swift — ILMessageFilterExtension implementation
- [x] Info.plist — extension configuration
- [ ] Verify Swift classifier matches Python predictions (parity test)

### 5D. Companion App (SwiftUI)
- [x] ContentView, MessageListView, MessageDetailView, TestClassifierView, SettingsView

### 5E. Xcode Project Setup
- [ ] Create Xcode project with two targets (app + extension)
- [ ] Configure App Group entitlement
- [ ] Build on physical device
- [ ] Test message filtering with real SMS

---

## Headline Results — LinearSVC, mean smishing F1, 30 seeds

| Strategy | Smishing F1 (std) | n_train (avg) |
|----------|------------------:|--------------:|
| manual_only | 0.870 (0.023) | 4,759 |
| synthetic_only | 0.933 (0.015) | 6,392 |
| **combined** | **0.939 (0.015)** | 11,151 |

- Best model: **LinearSVC + combined** (smishing F1 = 0.939, macro F1 = 0.949)
- Deployed at `outputs/models/best_model.joblib`
- Statistical: all pairwise comparisons significant (BH-corrected p < 0.05), Cohen's d 0.5 (small) to 2.65 (large)

## Notes

- Preprocessing uses `drop_exact_duplicates` (within-dataset deduplication only).
- No size matching: synthetic_only uses the full ~6,393 row training split.
- The synthetic dataset's ham class is 100% copied from manual ham — reported as a structural property of the synthetic corpus, not controlled for.
- Project framing: "given the natural output of each training pipeline, which produces the best classifier for SMS drawn from this real-world population?"
- Earlier multi-mode rerun + UCI integration + overlap-aware controls have been removed; their snapshot is at `outputs/runs/drop_exact_duplicates_original/` for project history.
