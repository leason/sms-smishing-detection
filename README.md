# Evaluating LLM-Generated Synthetic Datasets for SMS Smishing Detection

## Research Question

Can an LLM-generated synthetic SMS dataset train machine learning models that classify smishing better than models trained on a manually curated dataset?

## Datasets

| Dataset | File | Rows | Description |
|---------|------|-----:|-------------|
| Manual | `Dataset_5971.csv` | 5,971 | Manually curated real-world SMS messages |
| Synthetic | `Dataset_10191.csv` | 10,191 | LLM-generated SMS messages (using the manual dataset as source material) |

Both contain three classes: **ham**, **spam**, **smishing**. After within-dataset exact-text deduplication, the manual corpus has 5,949 unique rows and the synthetic corpus has 7,991. A provenance audit (see [`documentation/data-pipeline.md`](documentation/data-pipeline.md)) found that the synthetic dataset is structurally heterogeneous: 100% of its ham messages are verbatim copies of manual ham, while spam and smishing classes contain genuinely LLM-generated content.

## Experimental design

The project compares three training strategies, all evaluated against a randomized 20% holdout of the manual corpus (the real-world SMS test set):

| Strategy | Training data |
|----------|---------------|
| `manual_only` | Manual training split (~4,759 rows) |
| `synthetic_only` | Synthetic training split (~6,393 rows — full natural volume, no downsampling) |
| `combined` | Manual + synthetic training splits (~11,151 rows) |

Each strategy is run with three classifiers (Multinomial Naive Bayes, Logistic Regression, Linear SVM) across 30 random seeds = **270 total runs**. We embrace the natural volume produced by each pipeline rather than artificially size-matching, because volume is a core advantage of LLM-generated data and matching away that advantage would conceal what we're trying to measure.

## Headline Results

LinearSVC, mean smishing F1 (std) across 30 seeds, manual real-world holdout:

| Strategy | Smishing F1 |
|----------|------------:|
| `manual_only` | 0.870 (0.023) |
| `synthetic_only` | 0.933 (0.015) |
| `combined` | **0.939** (0.015) |

All three classifiers follow the same pattern: combined > synthetic_only > manual_only. Full per-classifier results in [`documentation/model-training.md`](documentation/model-training.md).

### Statistical analysis (pooled across classifiers)

| Comparison | Mean Δ | Adj. p | Cohen's d |
|-----------|-------:|-------:|----------:|
| `manual_only` vs `synthetic_only` | +0.045 | < 10⁻³⁰ | 2.18 (large) |
| `manual_only` vs `combined` | +0.058 | < 10⁻⁴⁰ | 2.65 (large) |
| `synthetic_only` vs `combined` | +0.011 | < 10⁻¹⁰ | 0.49 (small) |

All comparisons statistically significant (paired tests with Shapiro-Wilk gating, BH-corrected). The largest effect is between manual_only and either alternative — LLM-generated data clearly helps. The smaller `combined` vs `synthetic_only` margin (small-to-medium effect) indicates that adding real manual data on top of synthetic still helps but the marginal gain is modest at this volume.

### What the results show

- **Synthetic training outperforms manual training.** LLM-generated SMS data produces classifiers that score 6 percentage points higher on smishing F1 than classifiers trained on the manual corpus alone, with a large effect size.
- **Combined is the best strategy.** Combining the two corpora produces the strongest classifier (0.939 smishing F1, 0.949 macro F1), beating manual-only by 7 points and synthetic-only by 1 point.
- **Volume is part of the answer.** The synthetic corpus's natural advantage includes both volume (~6,393 vs ~4,759 training rows) and class balance (~33% smishing vs ~10%). Both are features of the LLM-generation pipeline, not confounds to control for.
- **Scope of the claim.** Results characterize performance on the test population (a randomized holdout of the manual corpus). The synthetic and manual corpora share ~70% of texts overall and 100% of ham messages, so this is not a test of generalization to unseen-content distributions — it is a test of which training pipeline produces a better classifier for SMS drawn from this real-world population.

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### R (for statistical analysis)

R 4.x with `tidyverse` and `effsize` packages.

## How to Run

### Training matrix

```bash
# 3 strategies × 3 models × 30 seeds = 270 runs, ~1.5 min
.venv/bin/python -m scripts.run_experiments

# Statistical analysis
Rscript r/statistical_analysis.R
```

Outputs land under `outputs/{metrics,figures,tables,models}/`. The best-performing model is serialized as `outputs/models/best_model.joblib` with metadata alongside.

### Jupyter notebook

The same training matrix is also executable via the notebook:

```bash
jupyter nbconvert --to html --execute notebooks/01_model_training_and_evaluation.ipynb
```

### FastAPI demo app (Docker)

The demo app serves the deployed model — LinearSVC trained on the `combined` strategy (smishing F1 = 0.939) — behind a small web UI and JSON API.

**Prerequisites**

- Docker Desktop installed and running ([install](https://www.docker.com/products/docker-desktop/))
- `outputs/models/best_model.joblib` exists. If it does not, generate it first by running the training matrix:

  ```bash
  .venv/bin/python -m scripts.run_experiments
  ```

**Build and run**

From the project root (`project/`):

```bash
docker compose up --build
```

The first build takes ~1–2 minutes (downloading `python:3.11-slim` and installing dependencies). On subsequent runs the cached image starts in seconds.

**Use the app**

Open **http://localhost:8000** in a browser. Paste an SMS message into the textarea (or click one of the six sample messages) and submit to see the predicted class (`ham`/`spam`/`smishing`), calibrated probabilities, and detected indicators (URLs, phone numbers, suspicious keywords). A footer badge shows the deployed-model metadata.

**Verify the service is healthy**

```bash
curl http://localhost:8000/health           # {"status":"ok","model_loaded":true}
curl http://localhost:8000/api/model        # deployed-model metadata
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Your account is locked. Verify at http://example.com"}'
```

**Stop**

```bash
docker compose down
```

**Swap in a freshly trained model**

`docker-compose.yml` mounts `./outputs/models` read-only into the container, so re-running the training matrix and then `docker compose restart` picks up a new `best_model.joblib` without rebuilding the image.

See [`documentation/demo-app.md`](documentation/demo-app.md) for endpoint reference, prediction flow, and known quirks.

## Project Structure

```
.
├── README.md
├── requirements.txt             # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── documentation/               # Technical docs
├── data/
│   ├── raw/                     # Source CSV files
│   └── processed/               # Cleaned datasets
├── notebooks/
│   └── 01_model_training_and_evaluation.ipynb
├── scripts/
│   └── run_experiments.py       # End-to-end runner
├── src/
│   ├── data/                    # preprocessing.py, audit.py
│   ├── features/                # text_features.py
│   ├── models/                  # train.py, evaluate.py, predict.py, figures.py
│   ├── app/                     # FastAPI app + templates
│   └── utils/paths.py           # Path constants
├── outputs/
│   ├── figures/                 # Generated plots (+ _data.csv sidecars)
│   ├── metrics/                 # Per-run results, confusion matrices
│   ├── models/                  # Serialized best model + metadata
│   ├── tables/                  # Audit and statistical tables
│   └── runs/drop_exact_duplicates_original/  # Frozen snapshot of initial run
└── r/
    └── statistical_analysis.R
```

## Outputs Summary

| Category | Key Files |
|----------|-----------|
| Metrics | `all_model_results.csv`, `confusion_matrices.csv`, `classification_reports.jsonl`, `train_composition.csv` |
| Tables | `dataset_audit.csv`, `cross_dataset_overlap.csv`, `leakage_audit.csv`, `statistical_tests.csv`, `effect_sizes.csv`, `model_performance_summary.csv`, `experiment_performance_summary.csv` |
| Figures | `class_balance_comparison.png`, `duplicate_rate_comparison.png`, `dataset_overlap_summary.png`, `message_length_distribution.png`, `mean_smishing_f1_by_training_strategy.png`, `mean_macro_f1_by_training_strategy.png`, `confusion_matrix_best_model_manual_holdout.png`, `smishing_f1_boxplot_by_training_strategy.png`, `statistical_comparison_boxplots.png` |
| Model | `best_model.joblib` (LinearSVC combined, F1=0.939), `best_model_metadata.json` |

## Documentation

- **[architecture.md](documentation/architecture.md)** — system design, module dependencies
- **[data-pipeline.md](documentation/data-pipeline.md)** — preprocessing, dataset audit, per-class overlap finding
- **[model-training.md](documentation/model-training.md)** — features, classifiers, experimental design, results
- **[statistical-analysis.md](documentation/statistical-analysis.md)** — R hypothesis tests and effect sizes
- **[demo-app.md](documentation/demo-app.md)** — FastAPI endpoints, Docker, prediction flow
- **[outputs-reference.md](documentation/outputs-reference.md)** — file catalog with schemas
- **[ios-app-proposal.md](documentation/ios-app-proposal.md)** — iOS SMS filtering feasibility study

## Limitations

- The synthetic dataset's ham subset is 100% copied from the manual corpus; only its spam and smishing examples are genuinely LLM-generated. Conclusions about "synthetic data" in this project are therefore specifically about LLM-generated spam/smishing combined with copied or augmented ham — not about purely LLM-generated SMS corpora.
- Evaluation is on a randomized holdout of the manual corpus. Because manual and synthetic share substantial text content, the holdout is not a test of generalization to unseen-content distributions; it is a test of in-distribution practical performance. Stronger generalization claims would require a third, independently sourced corpus.
- Only three linear/shallow classifiers were evaluated. Transformer-based models (e.g., BERT) may show different patterns.
- The manual dataset is heavily imbalanced (81% ham, 11% smishing, 8% spam). Stratified splitting and `class_weight="balanced"` mitigate this but do not eliminate it.
- The deployed model in the demo app is one realization of the best (model, strategy) combination, trained on seed=1. Headline metrics are averaged across 30 seeds.
- SMS messages evolve over time. Models trained on these static datasets may not generalize to future smishing patterns.
