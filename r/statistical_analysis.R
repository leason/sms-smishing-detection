#!/usr/bin/env Rscript
# ============================================================================
# Statistical Analysis: SMS Smishing Detection
# ============================================================================
#
# Reads per-seed metrics from the training matrix and runs paired hypothesis
# tests on the three training strategies (manual_only, synthetic_only,
# combined), evaluated against the manual real-world holdout.
#
# Test selection per comparison:
#   1. Compute paired differences (paired by seed).
#   2. Shapiro-Wilk normality test on the differences.
#   3. Paired t-test if p > 0.05, otherwise Wilcoxon signed-rank.
#   4. Benjamini-Hochberg FDR correction across all comparisons.
#   5. Cohen's d effect size on paired differences.
#
# Outputs:
#   outputs/tables/statistical_tests.csv
#   outputs/tables/effect_sizes.csv
#   outputs/figures/statistical_comparison_boxplots.png  (+ _data.csv sidecar)
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(effsize)
})

# Defaults point at the headline (drop_exact_duplicates) outputs. Override via
# env vars to run against an alternate matrix, e.g. the overlap_aware
# robustness check:
#   METRICS_PATH=outputs/runs/overlap_aware/metrics/all_model_results.csv \
#   TABLES_DIR=outputs/runs/overlap_aware/tables \
#   FIGURES_DIR=outputs/runs/overlap_aware/figures \
#   Rscript r/statistical_analysis.R
METRICS_PATH <- Sys.getenv("METRICS_PATH", "outputs/metrics/all_model_results.csv")
TABLES_DIR <- Sys.getenv("TABLES_DIR", "outputs/tables")
FIGURES_DIR <- Sys.getenv("FIGURES_DIR", "outputs/figures")

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

EXPERIMENT_ORDER <- c("manual_only", "synthetic_only", "combined")
MODEL_ORDER <- c("MultinomialNB", "LogisticRegression", "LinearSVC")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

paired_test <- function(a, b) {
  if (length(a) != length(b) || length(a) < 3) {
    return(tibble(
      test_type = NA_character_, n_pairs = length(a),
      mean_difference = NA_real_, median_difference = NA_real_,
      statistic = NA_real_, p_value = NA_real_, normality_p = NA_real_,
    ))
  }
  diffs <- b - a
  if (sd(diffs) == 0) {
    return(tibble(
      test_type = "degenerate_constant_diff", n_pairs = length(a),
      mean_difference = mean(diffs), median_difference = median(diffs),
      statistic = 0, p_value = 1, normality_p = NA_real_,
    ))
  }
  shap <- tryCatch(shapiro.test(diffs), error = function(e) list(p.value = NA))
  use_parametric <- !is.na(shap$p.value) && shap$p.value > 0.05
  if (use_parametric) {
    tt <- t.test(b, a, paired = TRUE)
    tibble(test_type = "paired_t_test", n_pairs = length(a),
           mean_difference = mean(diffs), median_difference = median(diffs),
           statistic = as.numeric(tt$statistic),
           p_value = tt$p.value, normality_p = shap$p.value)
  } else {
    wt <- wilcox.test(b, a, paired = TRUE, exact = FALSE)
    tibble(test_type = "wilcoxon_signed_rank", n_pairs = length(a),
           mean_difference = mean(diffs), median_difference = median(diffs),
           statistic = as.numeric(wt$statistic),
           p_value = wt$p.value, normality_p = shap$p.value)
  }
}

paired_effect_size <- function(a, b) {
  if (length(a) != length(b) || length(a) < 3) {
    return(tibble(cohens_d = NA_real_, cohens_d_magnitude = NA_character_,
                  wilcoxon_r = NA_real_))
  }
  diffs <- b - a
  if (sd(diffs) == 0) {
    return(tibble(cohens_d = 0, cohens_d_magnitude = "negligible", wilcoxon_r = 0))
  }
  cd <- tryCatch(cohen.d(b, a, paired = TRUE),
                 error = function(e) list(estimate = NA, magnitude = NA))
  wt <- wilcox.test(b, a, paired = TRUE, exact = FALSE)
  z_val <- qnorm(wt$p.value / 2) * sign(mean(diffs))
  tibble(
    cohens_d = if (is.list(cd)) cd$estimate else NA_real_,
    cohens_d_magnitude = if (is.list(cd)) as.character(cd$magnitude) else NA_character_,
    wilcoxon_r = abs(z_val) / sqrt(length(a)),
  )
}

paired_vector <- function(df, exp_id, model = NULL, metric) {
  d <- df %>% filter(experiment_id == exp_id)
  if (!is.null(model)) d <- d %>% filter(model_name == model)
  d %>% arrange(seed, model_name) %>% pull({{metric}})
}

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

metrics <- read_csv(METRICS_PATH, show_col_types = FALSE)
cat("Loaded ", nrow(metrics), " rows\n", sep = "")
cat("Models: ", paste(unique(metrics$model_name), collapse = ", "), "\n", sep = "")
cat("Experiments: ", paste(unique(metrics$experiment_id), collapse = ", "), "\n", sep = "")
cat("Seeds: ", length(unique(metrics$seed)), "\n\n", sep = "")

# ----------------------------------------------------------------------------
# Pairwise tests
# ----------------------------------------------------------------------------

comparisons <- combn(EXPERIMENT_ORDER, 2, simplify = FALSE)

results <- list()
efx <- list()

for (pair in comparisons) {
  exp_a <- pair[1]; exp_b <- pair[2]
  for (m in MODEL_ORDER) {
    a <- paired_vector(metrics, exp_a, m, smishing_f1)
    b <- paired_vector(metrics, exp_b, m, smishing_f1)
    if (length(a) == 0 || length(b) == 0) next
    results[[length(results) + 1]] <- bind_cols(
      tibble(comparison = paste(exp_a, "vs", exp_b),
             exp_a = exp_a, exp_b = exp_b, model_name = m),
      paired_test(a, b),
    )
    efx[[length(efx) + 1]] <- bind_cols(
      tibble(comparison = paste(exp_a, "vs", exp_b),
             exp_a = exp_a, exp_b = exp_b, model_name = m),
      paired_effect_size(a, b),
    )
  }
  # Pooled (ALL models)
  a_all <- paired_vector(metrics, exp_a, NULL, smishing_f1)
  b_all <- paired_vector(metrics, exp_b, NULL, smishing_f1)
  results[[length(results) + 1]] <- bind_cols(
    tibble(comparison = paste(exp_a, "vs", exp_b),
           exp_a = exp_a, exp_b = exp_b, model_name = "ALL"),
    paired_test(a_all, b_all),
  )
  efx[[length(efx) + 1]] <- bind_cols(
    tibble(comparison = paste(exp_a, "vs", exp_b),
           exp_a = exp_a, exp_b = exp_b, model_name = "ALL"),
    paired_effect_size(a_all, b_all),
  )
}

test_results <- bind_rows(results) %>%
  mutate(p_value_adjusted = p.adjust(p_value, method = "BH"),
         significant_at_0_05 = p_value_adjusted < 0.05)
effect_sizes <- bind_rows(efx)

cat("=== Pairwise Tests (smishing F1) ===\n")
print(test_results %>%
      select(comparison, model_name, mean_difference,
             p_value_adjusted, significant_at_0_05, test_type),
      n = 30)

cat("\n=== Effect Sizes ===\n")
print(effect_sizes %>%
      select(comparison, model_name, cohens_d, cohens_d_magnitude, wilcoxon_r),
      n = 30)

write_csv(test_results, file.path(TABLES_DIR, "statistical_tests.csv"))
write_csv(effect_sizes, file.path(TABLES_DIR, "effect_sizes.csv"))

# ----------------------------------------------------------------------------
# Figure: boxplot of smishing F1 by training strategy, grouped by model
# ----------------------------------------------------------------------------

plot_data <- metrics %>%
  mutate(
    experiment_id = factor(experiment_id, levels = EXPERIMENT_ORDER),
    model_name = factor(model_name, levels = MODEL_ORDER),
  )

p <- ggplot(plot_data,
            aes(x = experiment_id, y = smishing_f1, fill = model_name)) +
  geom_boxplot(alpha = 0.75, outlier.size = 1,
               position = position_dodge(width = 0.85)) +
  labs(
    title = "Smishing F1 Score by Training Strategy",
    subtitle = "Paired across 30 seeds, manual real-world holdout",
    x = "Training strategy",
    y = "Smishing F1",
    fill = "Classifier",
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom") +
  scale_fill_brewer(palette = "Set2")

ggsave(file.path(FIGURES_DIR, "statistical_comparison_boxplots.png"),
       p, width = 8, height = 5, dpi = 150)

write_csv(plot_data %>%
            select(seed, model_name, experiment_id, smishing_f1),
          file.path(FIGURES_DIR, "statistical_comparison_boxplots_data.csv"))

cat("\nDone. Outputs:\n")
cat("  ", file.path(TABLES_DIR, "statistical_tests.csv"), "\n")
cat("  ", file.path(TABLES_DIR, "effect_sizes.csv"), "\n")
cat("  ", file.path(FIGURES_DIR, "statistical_comparison_boxplots.png"), "\n")
