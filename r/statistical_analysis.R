#!/usr/bin/env Rscript
# Statistical Analysis: SMS Smishing Detection
# Paired hypothesis tests comparing training strategies
# ============================================================================

library(tidyverse)
library(effsize)

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------

metrics <- read_csv("outputs/metrics/all_model_results.csv", show_col_types = FALSE)

cat("Loaded", nrow(metrics), "rows\n")
cat("Models:", paste(unique(metrics$model_name), collapse = ", "), "\n")
cat("Experiments:", paste(unique(metrics$experiment_id), collapse = ", "), "\n")
cat("Seeds:", length(unique(metrics$seed)), "\n\n")

# ----------------------------------------------------------------------------
# 2. Define paired comparisons
# ----------------------------------------------------------------------------

comparisons <- tribble(
  ~comparison,                      ~exp_a,           ~exp_b,
  "manual_only vs synthetic_only",  "manual_only",    "synthetic_only",
  "manual_only vs combined",        "manual_only",    "combined",
  "synthetic_only vs combined",     "synthetic_only",  "combined",
)

# ----------------------------------------------------------------------------
# 3. Run paired tests (per model and overall)
# ----------------------------------------------------------------------------

run_paired_tests <- function(metrics, comparisons) {
  results <- list()

  for (i in seq_len(nrow(comparisons))) {
    comp <- comparisons[i, ]

    # Per-model tests
    for (model in unique(metrics$model_name)) {
      a <- metrics %>%
        filter(experiment_id == comp$exp_a, model_name == model) %>%
        arrange(seed) %>%
        pull(smishing_f1)
      b <- metrics %>%
        filter(experiment_id == comp$exp_b, model_name == model) %>%
        arrange(seed) %>%
        pull(smishing_f1)

      if (length(a) != length(b) || length(a) < 3) next

      diffs <- b - a

      # Normality test on paired differences
      shap <- shapiro.test(diffs)
      use_parametric <- shap$p.value > 0.05

      if (use_parametric) {
        tt <- t.test(b, a, paired = TRUE)
        test_type <- "paired_t_test"
        statistic <- tt$statistic
        p_value <- tt$p.value
      } else {
        wt <- wilcox.test(b, a, paired = TRUE, exact = FALSE)
        test_type <- "wilcoxon_signed_rank"
        statistic <- wt$statistic
        p_value <- wt$p.value
      }

      results[[length(results) + 1]] <- tibble(
        comparison = comp$comparison,
        metric = "smishing_f1",
        model_name = model,
        test_type = test_type,
        n_pairs = length(a),
        mean_difference = mean(diffs),
        median_difference = median(diffs),
        statistic = as.numeric(statistic),
        p_value = p_value,
        normality_p = shap$p.value,
      )
    }

    # Overall test (all models pooled, paired by seed + model)
    a_all <- metrics %>%
      filter(experiment_id == comp$exp_a) %>%
      arrange(seed, model_name) %>%
      pull(smishing_f1)
    b_all <- metrics %>%
      filter(experiment_id == comp$exp_b) %>%
      arrange(seed, model_name) %>%
      pull(smishing_f1)

    if (length(a_all) == length(b_all) && length(a_all) >= 3) {
      diffs_all <- b_all - a_all
      shap_all <- shapiro.test(diffs_all)
      use_parametric_all <- shap_all$p.value > 0.05

      if (use_parametric_all) {
        tt_all <- t.test(b_all, a_all, paired = TRUE)
        test_type_all <- "paired_t_test"
        stat_all <- tt_all$statistic
        p_all <- tt_all$p.value
      } else {
        wt_all <- wilcox.test(b_all, a_all, paired = TRUE, exact = FALSE)
        test_type_all <- "wilcoxon_signed_rank"
        stat_all <- wt_all$statistic
        p_all <- wt_all$p.value
      }

      results[[length(results) + 1]] <- tibble(
        comparison = comp$comparison,
        metric = "smishing_f1",
        model_name = "ALL",
        test_type = test_type_all,
        n_pairs = length(a_all),
        mean_difference = mean(diffs_all),
        median_difference = median(diffs_all),
        statistic = as.numeric(stat_all),
        p_value = p_all,
        normality_p = shap_all$p.value,
      )
    }
  }

  bind_rows(results)
}

test_results <- run_paired_tests(metrics, comparisons)

# Apply Benjamini-Hochberg FDR correction
test_results <- test_results %>%
  mutate(
    p_value_adjusted = p.adjust(p_value, method = "BH"),
    significant_at_0_05 = p_value_adjusted < 0.05
  )

cat("=== Statistical Tests ===\n")
print(test_results %>% select(comparison, model_name, test_type, mean_difference, p_value, p_value_adjusted, significant_at_0_05), n = 20)

# ----------------------------------------------------------------------------
# 4. Effect sizes
# ----------------------------------------------------------------------------

compute_effect_sizes <- function(metrics, comparisons) {
  results <- list()

  for (i in seq_len(nrow(comparisons))) {
    comp <- comparisons[i, ]

    for (model in c(unique(metrics$model_name), "ALL")) {
      if (model == "ALL") {
        a <- metrics %>% filter(experiment_id == comp$exp_a) %>% arrange(seed, model_name) %>% pull(smishing_f1)
        b <- metrics %>% filter(experiment_id == comp$exp_b) %>% arrange(seed, model_name) %>% pull(smishing_f1)
      } else {
        a <- metrics %>% filter(experiment_id == comp$exp_a, model_name == model) %>% arrange(seed) %>% pull(smishing_f1)
        b <- metrics %>% filter(experiment_id == comp$exp_b, model_name == model) %>% arrange(seed) %>% pull(smishing_f1)
      }

      if (length(a) != length(b) || length(a) < 3) next

      # Cohen's d for paired differences
      cd <- cohen.d(b, a, paired = TRUE)

      # Wilcoxon r = Z / sqrt(N)
      wt <- wilcox.test(b, a, paired = TRUE, exact = FALSE)
      z_val <- qnorm(wt$p.value / 2) * sign(mean(b - a))
      wilcox_r <- abs(z_val) / sqrt(length(a))

      results[[length(results) + 1]] <- tibble(
        comparison = comp$comparison,
        metric = "smishing_f1",
        model_name = model,
        cohens_d = cd$estimate,
        cohens_d_magnitude = cd$magnitude,
        wilcoxon_r = wilcox_r,
      )
    }
  }

  bind_rows(results)
}

effect_sizes <- compute_effect_sizes(metrics, comparisons)

cat("\n=== Effect Sizes ===\n")
print(effect_sizes, n = 20)

# ----------------------------------------------------------------------------
# 5. Save outputs
# ----------------------------------------------------------------------------

dir.create("outputs/tables", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs/figures", showWarnings = FALSE, recursive = TRUE)

write_csv(test_results, "outputs/tables/statistical_tests.csv")
write_csv(effect_sizes, "outputs/tables/effect_sizes.csv")

cat("\nSaved: outputs/tables/statistical_tests.csv\n")
cat("Saved: outputs/tables/effect_sizes.csv\n")

# ----------------------------------------------------------------------------
# 6. Boxplot figure
# ----------------------------------------------------------------------------

plot_data <- metrics %>%
  mutate(experiment_id = factor(experiment_id,
    levels = c("manual_only", "synthetic_only", "combined")))

p <- ggplot(plot_data, aes(x = experiment_id, y = smishing_f1, fill = model_name)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  labs(
    title = "Smishing F1 Score by Training Strategy",
    subtitle = "Paired across 30 seeds, evaluated on Manual Holdout",
    x = "Training Strategy",
    y = "Smishing F1",
    fill = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom") +
  scale_fill_brewer(palette = "Set2")

ggsave("outputs/figures/statistical_comparison_boxplots.png", p,
  width = 8, height = 5, dpi = 150)

cat("Saved: outputs/figures/statistical_comparison_boxplots.png\n")
cat("\nDone.\n")
