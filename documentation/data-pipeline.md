# Data Pipeline

## Source: `src/data/preprocessing.py`, `src/data/audit.py`

## Datasets

| Dataset | File | Rows | Source |
|---------|------|-----:|--------|
| Manual | `data/raw/Dataset_5971.csv` | 5,971 | Manually curated real-world SMS |
| Synthetic | `data/raw/Dataset_10191.csv` | 10,191 | LLM-generated from manual dataset |

### Provenance

The synthetic dataset was generated using the manual dataset as source material. After deduplication, approximately 70% of manual texts appear verbatim in the synthetic dataset. This has significant implications for experimental design (see [model-training.md](model-training.md)).

### Raw Schema

Both CSVs share the same columns:

| Raw Column | Type | Content |
|------------|------|---------|
| `LABEL` | string | Class label (inconsistent casing) |
| `TEXT` | string | SMS message body |
| `URL` | string | "Yes"/"No"/"yes"/"No" — contains URL |
| `EMAIL` | string | Same — contains email |
| `PHONE` | string | Same — contains phone number |

### Raw Label Issues

`Dataset_5971.csv` has mixed-case labels: `Smishing`/`smishing`, `Spam`/`spam`. The synthetic dataset uses consistent lowercase.

## Preprocessing Steps

All logic lives in `src/data/preprocessing.py`. Entry point: `run_preprocessing(mode, save)`.

### 1. Column Normalization

```
LABEL → label
TEXT  → text
URL   → has_url
EMAIL → has_email
PHONE → has_phone
```

### 2. Label Normalization

```
ham      → ham
spam     → spam
Spam     → spam
smishing → smishing
Smishing → smishing
```

Any value outside this map raises a `ValueError`.

### 3. Indicator Normalization

`has_url`, `has_email`, `has_phone` are converted to binary integers (0/1). Recognized truthy values: `yes`, `y`, `true`, `1`. Recognized falsy: `no`, `n`, `false`, `0`. Unrecognized values emit a warning and default to 0.

### 4. Text Cleaning

The `text_clean` column is derived from `text`:
- Convert to string
- Strip leading/trailing whitespace
- Collapse internal whitespace to single spaces
- Preserves casing (TF-IDF vectorizer handles lowercasing)
- Preserves URLs, phone numbers, email addresses (they are predictive features)

### 5. Validation

Checks for:
- Missing `text` or `label` values (raises `ValueError`)
- Labels outside `{ham, spam, smishing}` (raises `ValueError`)

## Duplicate and Overlap Handling

Three modes, controlled by the `mode` parameter:

| Mode | Behavior |
|------|----------|
| `keep_duplicates` | No deduplication. Used for sensitivity analysis. |
| `drop_exact_duplicates` | Drop duplicates within each dataset based on raw `text` column, keeping first occurrence. **This is the primary analysis mode.** |
| `overlap_aware` | `drop_exact_duplicates` + remove synthetic rows whose text appears in the manual dataset. |

Deduplication uses raw `text` (not `text_clean`) because whitespace differences may be meaningful.

### Post-Dedup Counts (drop_exact_duplicates mode)

| Dataset | Rows | ham | spam | smishing |
|---------|-----:|----:|-----:|---------:|
| Manual | 5,949 | 4,834 | 487 | 628 |
| Synthetic | 7,991 | 3,393 | 1,944 | 2,654 |

## Dataset Audit

Source: `src/data/audit.py`. Entry point: `run_audit(manual, synthetic, save)`.

### Audit Tables

| File | Content |
|------|---------|
| `outputs/tables/dataset_audit.csv` | Row counts, class balance, duplicate count/%, avg message length, vocabulary size |
| `outputs/tables/cross_dataset_overlap.csv` | Every synthetic text that also appears in manual |
| `outputs/tables/top_duplicate_messages.csv` | Top 20 most duplicated messages across both datasets |
| `outputs/tables/leakage_audit.csv` | Overlap counts and percentages |

### Audit Figures

| File | Content |
|------|---------|
| `class_balance_comparison.png` | Side-by-side bar charts of class distribution |
| `duplicate_rate_comparison.png` | Duplicate percentage per dataset |
| `dataset_overlap_summary.png` | Overlap percentage bars |
| `message_length_distribution.png` | Overlaid histograms of message length |

### Key Audit Findings

- Manual dataset: 0% internal duplicates after dedup (22 removed)
- Synthetic dataset: 0% after dedup (2,200 removed)
- Cross-dataset overlap: 4,169 texts (70% of manual, 52% of synthetic)
- Manual avg message length: 83 chars; Synthetic: 113 chars
- Manual vocabulary: 14,883 unique words; Synthetic: 12,228

## Output Files

Preprocessing saves three CSVs to `data/processed/`:

| File | Content |
|------|---------|
| `manual_clean.csv` | Processed manual dataset |
| `synthetic_clean.csv` | Processed synthetic dataset |
| `combined_clean.csv` | Concatenation of both |

Each contains columns: `dataset_source`, `label`, `text`, `text_clean`, `has_url`, `has_email`, `has_phone`.
