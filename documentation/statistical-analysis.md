# Statistical Analysis

## Source: `r/statistical_analysis.R`

## Purpose

Formal hypothesis testing on the per-seed metrics from the training matrix. Determines whether differences between training strategies are statistically significant and practically meaningful.

## Input

```
outputs/metrics/all_model_results.csv
```

270 rows (30 seeds × 3 models × 3 strategies), each with per-run metrics including `smishing_f1`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `tidyverse` | Data manipulation and plotting |
| `effsize` | Cohen's d effect size calculation |

Base R provides `t.test()`, `wilcox.test()`, `shapiro.test()`, `p.adjust()`.

## Comparisons

Three pairwise comparisons, each run per-model and pooled across all models:

| Comparison | Question |
|-----------|---------|
| `manual_only` vs `synthetic_only` | Does training source matter? |
| `manual_only` vs `combined` | Does adding synthetic data help over manual? |
| `synthetic_only` vs `combined` | Does adding real data help over synthetic? |

12 test rows total: 3 comparisons × (3 classifiers + 1 pooled).

## Test Selection

For each comparison:

1. Compute paired differences (paired by `seed`; for pooled-ALL rows, paired by `seed × model_name`).
2. Shapiro-Wilk normality test on the differences.
3. If p > 0.05: paired t-test (parametric).
4. Otherwise: Wilcoxon signed-rank (non-parametric).
5. All tests are two-tailed.

## Multiple Comparison Correction

**Benjamini-Hochberg FDR correction** is applied across all 12 tests.

## Effect Sizes

| Metric | Computation |
|--------|-------------|
| Cohen's d (paired) | `effsize::cohen.d(b, a, paired=TRUE)` |
| Wilcoxon r | `|Z| / sqrt(N)` derived from Wilcoxon p-value |

### Interpretation scale (Cohen's d)

| \|d\| | Magnitude |
|-------|-----------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| ≥ 0.8 | Large |

## Results

Pooled across the three classifiers (smishing F1):

| Comparison | Mean Δ | Adj. p | Cohen's d |
|-----------|-------:|-------:|----------:|
| `manual_only` vs `synthetic_only` | +0.045 | < 10⁻³⁰ | 2.18 (large) |
| `manual_only` vs `combined` | +0.058 | < 10⁻⁴⁰ | 2.65 (large) |
| `synthetic_only` vs `combined` | +0.011 | < 10⁻¹⁰ | 0.49 (small) |

All 12 tests (3 comparisons × 4 groupings) are statistically significant after BH correction. Per-classifier breakdowns and exact statistics are in `outputs/tables/statistical_tests.csv` and `outputs/tables/effect_sizes.csv`.

### Reading these results

- **The largest effect is between manual_only and either synthetic-using strategy.** This tells us that the manual corpus alone is meaningfully behind anything that uses synthetic data, with consistently large effect sizes across all three classifiers.
- **The `synthetic_only` vs `combined` comparison is a small effect for LinearSVC and Multinomial NB but large for Logistic Regression.** Adding real ham/spam/smishing on top of the synthetic corpus helps Logistic Regression noticeably; for the other classifiers the marginal gain is small. All differences remain statistically significant.
- **Manual_only baseline is the same across all comparisons** because manual data is unchanged regardless of which alternative it's compared against. This serves as an internal consistency check.

## Output Files

| File | Content |
|------|---------|
| `outputs/tables/statistical_tests.csv` | All test results: comparison, model, test_type, n_pairs, mean_difference, statistic, p_value, normality_p, p_value_adjusted, significant_at_0_05 |
| `outputs/tables/effect_sizes.csv` | comparison, model, cohens_d, cohens_d_magnitude, wilcoxon_r |
| `outputs/figures/statistical_comparison_boxplots.png` | ggplot2 boxplot of smishing F1 by strategy, grouped by classifier |
| `outputs/figures/statistical_comparison_boxplots_data.csv` | Source data for the boxplot |

## Running

```bash
Rscript r/statistical_analysis.R
```

Prints a summary and saves all output files. Takes a few seconds.
