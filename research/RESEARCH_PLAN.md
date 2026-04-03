# Research Implementation Plan

This document turns the current engineering project into a hypothesis-driven research workflow.

## 1. Research Questions and Hypotheses

RQ1: Does SSL+MMViT outperform classical and CNN baselines on the same fixed test split?
H1: SSL+MMViT has higher Macro-F1 than SVM, ResNet-50, and DenseNet-121.

RQ2: Which classes remain the primary failure modes across model families?
H2: Confusions are concentrated in visually similar recyclable classes.

RQ3: How stable are results across repeated runs?
H3: Deep models show low variance in Macro-F1 under fixed protocol.

## 2. Primary Outcome Metrics

- Macro-F1 (primary metric)
- Accuracy
- Weighted-F1
- Per-class Precision/Recall/F1

## 3. Standardized Protocol

1. Create fixed train/val/test split once using seed=42.
2. Train each baseline on identical split.
3. Evaluate SSL+MMViT on the identical shared test split.
4. Compare all models using the same metric definitions.
5. Repeat complete run multiple times and report mean +/- std.

## 4. Reproducible Commands

From project root:

- Quick run:
  - python research/run_research_pipeline.py --mode quick
- Full run:
  - python research/run_research_pipeline.py --mode full
- Full multi-run statistics (example N=3):
  - python research/run_research_pipeline.py --mode full --trials 3
- Recompute comparison only (no retraining):
  - python research/run_research_pipeline.py --skip-train

Outputs for each run are archived in:

- research/runs/<run_tag>/trial_XX_metrics.json
- research/runs/<run_tag>/aggregate_stats.json
- research/runs/<run_tag>/paper_results_table.md
- research/runs/<run_tag>/paper_results_table.csv

## 5. Minimum Evidence for a Research Report

- Table: model-wise Accuracy, Macro-F1, Weighted-F1
- Table: per-class F1 across all models
- Confusion matrices for best and second-best models
- Error analysis with 10 to 20 representative failure samples
- Limitations and threats to validity section

## 6. Statistical Reporting Guidance

For N repeated complete runs per model:

- Mean metric: mu = (1/N) * sum(x_i)
- Sample std: s = sqrt((1/(N-1)) * sum((x_i - mu)^2))
- 95% CI (approx): mu +/- 1.96 * s / sqrt(N)

Use the same N for all compared models.

The pipeline computes and exports mean, sample std, and 95% CI automatically.
