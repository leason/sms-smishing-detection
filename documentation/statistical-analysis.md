# Statistical Analysis

## Source: `r/statistical_analysis.R`

## Purpose

The R script performs formal hypothesis testing to determine whether differences in smishing detection performance across training strategies are statistically significant and practically meaningful.

## Input

```
outputs/metrics/all_model_results.csv
```

270 rows (30 seeds x 3 models x 3 experiments), each with per-run metrics.

## Dependencies

| Package | Purpose |
|---------|---------|
| `tidyverse` | Data manipulation and plotting |
| `effsize` | Cohen's d effect size calculation |

Base R provides `t.test()`, `wilcox.test()`, `shapiro.test()`, and `p.adjust()`.

## Comparisons

Three pairwise comparisons, each run per-model and pooled across all models:

| Comparison | Question |
|------------|----------|
| `manual_only` vs `synthetic_only` | Does training data source matter? |
| `manual_only` vs `combined` | Does adding synthetic data help? |
| `synthetic_only` vs `combined` | Does adding real data help? |

## Test Selection

For each comparison:

1. Compute paired differences (paired by seed + model_name)
2. Run Shapiro-Wilk normality test on differences
3. If p > 0.05: use **paired t-test** (parametric)
4. If p <= 0.05: use **Wilcoxon signed-rank test** (non-parametric)
5. All tests are **two-tailed**

In practice, 11 of 12 comparisons used paired t-tests; only LinearSVC manual_only vs synthetic_only used Wilcoxon.

## Multiple Comparison Correction

**Benjamini-Hochberg FDR correction** applied across all 12 tests (3 comparisons x 4 groupings: 3 per-model + 1 pooled).

## Effect Sizes

| Metric | When Used |
|--------|-----------|
| Cohen's d (paired) | Always computed via `effsize::cohen.d(b, a, paired=TRUE)` |
| Wilcoxon r | Always computed as `|Z| / sqrt(N)` where Z is derived from the Wilcoxon p-value |

### Interpretation Scale (Cohen's d)

| d | Magnitude |
|---|-----------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

## Results

All 12 comparisons are statistically significant (BH-adjusted p < 0.05).

### Key Findings

**manual_only vs synthetic_only:**
- Synthetic outperforms manual by 3–6 percentage points in smishing F1
- Effect sizes: large (Cohen's d 1.6–2.9)
- Even with size-matched training data, synthetic data trains better smishing detectors

**manual_only vs combined:**
- Combined outperforms manual by 5–7 percentage points
- Effect sizes: large (d 2.5–3.4)

**synthetic_only vs combined:**
- Combined outperforms synthetic-only by 1–2.5 percentage points
- Effect sizes: medium to large (d 0.6–1.5)
- Adding real data on top of synthetic still helps, but the marginal gain is smaller

## Output Files

| File | Content |
|------|---------|
| `outputs/tables/statistical_tests.csv` | All test results with columns: comparison, metric, model_name, test_type, n_pairs, mean_difference, median_difference, statistic, p_value, normality_p, p_value_adjusted, significant_at_0_05 |
| `outputs/tables/effect_sizes.csv` | Effect sizes with columns: comparison, metric, model_name, cohens_d, cohens_d_magnitude, wilcoxon_r |
| `outputs/figures/statistical_comparison_boxplots.png` | ggplot2 boxplot of smishing F1 by training strategy, grouped by model |

## Running

```bash
Rscript r/statistical_analysis.R
```

Prints results to console and saves all output files. Takes a few seconds.
