# Results Summary — AIMCON_JMI_26

Auto-generated from experiment outputs. Update after running `bash experiments/run_all.sh`.

---

## Experiment 9 — Metadata-Label Correlation
**File:** `outputs/tables/exp09_correlation_statistics.csv`

| Test | Statistic | p-value | Effect Size (Cramér's V) |
|------|-----------|---------|--------------------------|
| Chi² sex × class | — | — | — |
| Chi² site × class | — | — | — |
| Kruskal-Wallis age × class | — | — | — |
| Point-biserial age × malignant | — | — | — |

---

## Experiment 1 — Baseline Models
**File:** `outputs/tables/exp01_baseline_metrics.csv`

| Model | Balanced Accuracy | Macro AUC | Macro F1 |
|-------|------------------|-----------|----------|
| image_only | — | — | — |
| metadata_only | — | — | — |
| fusion | — | — | — |

---

## Experiment 2 — Modality Occlusion
**File:** `outputs/tables/exp02_occlusion_results.csv`

| Condition | Balanced Accuracy | Drop vs Full |
|-----------|------------------|--------------|
| Full | — | — |
| Occlude Image | — | — |
| Occlude Metadata | — | — |

---

## Experiment 4 — Subgroup Analysis
**File:** `outputs/tables/exp04_subgroup_metrics.csv`

| Model | Group | MEL FNR | BCC FNR | SCC FNR |
|-------|-------|---------|---------|---------|
| fusion | indian_profile | — | — | — |
| fusion | western_profile | — | — | — |
| image_only | indian_profile | — | — | — |
| image_only | western_profile | — | — | — |

---

## Experiments 5 & 6 — Mitigations
**File:** `outputs/tables/exp05_dropout_comparison.csv`, `exp06_reweighting_comparison.csv`

*Populate after training.*

---

## Key Findings (to populate after run)

1. **Shortcut exists:** metadata-only AUC = X.XX vs image-only AUC = X.XX (gap = X.XX)
2. **Fusion is metadata-dominated:** occluding metadata drops balanced accuracy by X%, occluding image by Y%
3. **Subgroup gap:** MEL FNR for indian_profile = X.XX vs western_profile = X.XX (gap = X.XX)
4. **Mitigation:** metadata dropout reduces subgroup FNR gap from X.XX to X.XX
