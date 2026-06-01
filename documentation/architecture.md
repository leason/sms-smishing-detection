# Architecture

## Overview

The project has three independent workstreams that share data through the filesystem:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (data/)                        │
│  raw CSVs → preprocessing → processed CSVs                  │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
       ┌───────▼────────┐    ┌───────▼────────┐
       │  ML Pipeline    │    │  Dataset Audit  │
       │  (Python)       │    │  (Python)       │
       │                 │    │                 │
       │  Train models   │    │  Overlap,       │
       │  Evaluate        │    │  duplicates,    │
       │  Select best    │    │  leakage        │
       └───────┬────────┘    └───────┬────────┘
               │                      │
       ┌───────▼──────────────────────▼───────┐
       │         Outputs (outputs/)            │
       │  metrics/ figures/ tables/ models/    │
       └───────┬──────────────────────┬───────┘
               │                      │
       ┌───────▼────────┐    ┌───────▼────────┐
       │  R Statistical  │    │  FastAPI App    │
       │  Analysis       │    │  (Docker)       │
       │                 │    │                 │
       │  Reads metrics  │    │  Loads best     │
       │  CSV, produces  │    │  model .joblib  │
       │  test results   │    │  for inference  │
       └────────────────┘    └────────────────┘
```

## Module Dependency Graph

```
src/utils/paths.py          ← No dependencies. All path constants.
    │
    ├── src/data/preprocessing.py   ← Loading, normalization, dedup
    │       │
    │       └── src/data/audit.py   ← Provenance & leakage analysis
    │
    ├── src/features/text_features.py  ← TF-IDF + indicator pipeline (no src deps)
    │       │
    │       └── src/models/train.py    ← Experiment loop, model selection
    │               │
    │               ├── src/models/evaluate.py  ← Metrics extraction, CSV output
    │               │
    │               └── src/models/figures.py   ← Evaluation visualizations
    │
    └── src/models/predict.py   ← Inference utilities (no src deps)
            │
            └── src/app/main.py ← FastAPI application
```

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| ML pipeline | Python, scikit-learn, pandas, numpy | Python 3.11, sklearn 1.x |
| Feature extraction | TF-IDF (sklearn), ColumnTransformer | — |
| Statistical analysis | R, tidyverse, effsize | R 4.5 |
| Demo app | FastAPI, Jinja2, Uvicorn | — |
| Containerization | Docker, docker-compose | — |
| Model serialization | joblib | — |
| Visualization | matplotlib, seaborn, ggplot2 (R) | — |

## Data Flow

1. **Raw CSVs** (`data/raw/`) are loaded by `preprocessing.py`
2. **Preprocessing** normalizes labels, indicators, cleans text, handles duplicates
3. **Processed CSVs** saved to `data/processed/`
4. **Audit** runs overlap and leakage analysis, saves tables and figures to `outputs/`
5. **Training loop** creates splits, trains 3 models x 3 strategies x 30 seeds
6. **Evaluation** extracts metrics per run, saves to `outputs/metrics/`
7. **Best model** selected by mean smishing F1, serialized to `outputs/models/`
8. **R script** reads `all_model_results.csv`, runs paired tests, saves to `outputs/tables/`
9. **FastAPI app** loads `best_model.joblib` at startup, serves predictions

## Key Architectural Decisions

- **scikit-learn Pipelines**: Every model is a single Pipeline object (TF-IDF + indicators + classifier). This means the serialized `.joblib` contains everything needed for inference — no separate tokenizer or feature config.
- **CalibratedClassifierCV**: LinearSVC doesn't natively support `predict_proba()`. Wrapping it in `CalibratedClassifierCV` enables probability output for the demo app.
- **Filesystem as interface**: The three workstreams (Python ML, R stats, FastAPI app) communicate through CSV files and the serialized model, with no database or message queue.
- **Reproducibility through seeds**: All randomness is controlled via explicit seeds 1–30. The same seed produces the same splits and training results.
