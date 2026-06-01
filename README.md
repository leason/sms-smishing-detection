# Evaluating LLM-Generated Synthetic Datasets for SMS Smishing Detection

## Research Question

Can an LLM-generated synthetic SMS dataset be used to effectively train machine learning models for smishing detection on real-world SMS messages?

## Datasets

| Dataset | File | Rows | Description |
|---------|------|-----:|-------------|
| Manual | `Dataset_5971.csv` | 5,971 | Manually curated real-world SMS messages |
| Synthetic | `Dataset_10191.csv` | 10,191 | LLM-generated SMS messages |

The synthetic dataset was generated using the manual dataset as source material. Approximately 70% of manual texts appear in the synthetic dataset, which the pipeline accounts for through duplicate removal and leakage auditing.

Both datasets contain three classes: **ham**, **spam**, and **smishing**.

## Key Results

All experiments are evaluated against a manual (real-world) holdout set across 30 random seeds. The `synthetic_only` experiment uses size-matched training data to control for dataset size differences.

| Model | manual_only | synthetic_only | combined |
|-------|-------------|----------------|----------|
| LinearSVC | 0.870 | 0.930 | **0.939** |
| Logistic Regression | 0.868 | 0.900 | 0.925 |
| Multinomial NB | 0.857 | 0.887 | 0.906 |

*Values are mean smishing-class F1 scores.*

All pairwise differences are statistically significant (paired t-tests, BH-corrected p < 0.05) with large effect sizes (Cohen's d 0.6–3.4). Synthetic training data consistently outperforms manual-only training, and combining both sources yields the best performance.

## Setup

### Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### R (for statistical analysis)

R 4.x with `tidyverse` and `effsize` packages installed.

## How to Run

### 1. Preprocessing, Training, and Evaluation

Run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_model_training_and_evaluation.ipynb
```

Or execute it non-interactively:

```bash
jupyter nbconvert --to html --execute notebooks/01_model_training_and_evaluation.ipynb
```

This will:
- Load and preprocess both datasets
- Run the dataset provenance audit
- Train 3 models x 3 strategies x 30 seeds (270 runs)
- Save all metrics, figures, tables, and the best serialized model

### 2. Statistical Analysis

```bash
Rscript r/statistical_analysis.R
```

Reads `outputs/metrics/all_model_results.csv` and produces paired hypothesis tests, effect sizes, and comparison boxplots.

### 3. FastAPI Demo App

```bash
docker compose up --build
```

Then open **http://localhost:8000** to classify SMS messages through the web interface.

Stop with:

```bash
docker compose down
```

## Project Structure

```
.
├── README.md
├── requirements.md              # Detailed project requirements
├── requirements.txt             # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── documentation/               # Technical docs (architecture, pipeline, etc.)
├── data/
│   ├── raw/                     # Source CSV files
│   └── processed/               # Cleaned datasets
├── notebooks/
│   └── 01_model_training_and_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Loading, normalization, dedup
│   │   └── audit.py             # Provenance & leakage analysis
│   ├── features/
│   │   └── text_features.py     # TF-IDF + indicator pipeline
│   ├── models/
│   │   ├── train.py             # Experiment loop & model selection
│   │   ├── evaluate.py          # Metrics extraction & summaries
│   │   ├── predict.py           # Inference utilities
│   │   └── figures.py           # Evaluation visualizations
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   └── templates/
│   │       └── index.html       # Web UI
│   └── utils/
│       └── paths.py             # Project path constants
├── outputs/
│   ├── figures/                  # All generated plots
│   ├── metrics/                  # Per-run results, confusion matrices
│   ├── models/                   # Serialized best model + metadata
│   └── tables/                   # Audit, summary, and statistical tables
└── r/
    └── statistical_analysis.R    # Hypothesis testing & effect sizes
```

## Outputs Summary

| Category | Key Files |
|----------|-----------|
| Metrics | `all_model_results.csv`, `confusion_matrices.csv`, `classification_reports.jsonl` |
| Tables | `dataset_audit.csv`, `leakage_audit.csv`, `statistical_tests.csv`, `effect_sizes.csv` |
| Figures | Class balance, duplicate rates, overlap, F1 comparisons, confusion matrices, boxplots |
| Model | `best_model.joblib` (LinearSVC combined, F1=0.939), `best_model_metadata.json` |

## Documentation

See the [`documentation/`](documentation/) folder for detailed technical docs:

- **[architecture.md](documentation/architecture.md)** — system design, module dependencies, data flow
- **[data-pipeline.md](documentation/data-pipeline.md)** — preprocessing, audit, overlap analysis
- **[model-training.md](documentation/model-training.md)** — features, classifiers, experimental design
- **[statistical-analysis.md](documentation/statistical-analysis.md)** — R hypothesis tests and effect sizes
- **[demo-app.md](documentation/demo-app.md)** — FastAPI endpoints, Docker, prediction flow
- **[outputs-reference.md](documentation/outputs-reference.md)** — complete file catalog with schemas
- **[ios-app-proposal.md](documentation/ios-app-proposal.md)** — iOS SMS filtering feasibility study

## Limitations

- The synthetic dataset was derived from the manual dataset, creating ~70% text overlap. While the pipeline audits and controls for this, the datasets are not truly independent.
- Evaluation is limited to three linear/shallow classifiers. Deep learning models (e.g., BERT) are not explored.
- The manual dataset is heavily imbalanced (81% ham, 11% smishing, 8% spam). Stratified splitting and `class_weight="balanced"` mitigate this but do not eliminate it.
- The demo app uses a model trained on a single seed (seed=1). Performance metrics are averaged across 30 seeds but the deployed model is one instance.
- SMS messages evolve over time. Models trained on these static datasets may not generalize to future smishing patterns.
