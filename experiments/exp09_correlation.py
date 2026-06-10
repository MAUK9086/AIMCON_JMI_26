"""Experiment 9 — Metadata-Label Correlation Analysis (no training required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy import stats


def main(cfg: dict) -> None:
    print("[EXP09] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from data.preprocessing import load_and_merge, impute_metadata, CLASS_NAMES
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    df["class_name"] = df["label"].map({i: c for i, c in enumerate(CLASS_NAMES)})
    df["malignant"] = df["class_name"].isin(["MEL", "BCC", "SCC"]).astype(int)

    stat_rows = []

    # 1. Chi-squared: sex vs class
    ct_sex = pd.crosstab(df["sex"], df["class_name"])
    chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(ct_sex)
    n = ct_sex.values.sum()
    cramer_sex = np.sqrt(chi2_sex / (n * (min(ct_sex.shape) - 1)))
    stat_rows.append({"test": "chi2_sex_class", "statistic": chi2_sex, "p_value": p_sex, "effect_size_cramers_v": cramer_sex, "dof": dof_sex})

    # 2. Chi-squared: site vs class
    ct_site = pd.crosstab(df["anatom_site_general"], df["class_name"])
    chi2_site, p_site, dof_site, _ = stats.chi2_contingency(ct_site)
    cramer_site = np.sqrt(chi2_site / (n * (min(ct_site.shape) - 1)))
    stat_rows.append({"test": "chi2_site_class", "statistic": chi2_site, "p_value": p_site, "effect_size_cramers_v": cramer_site, "dof": dof_site})

    # 3. Kruskal-Wallis: age across classes
    groups = [df[df["class_name"] == c]["age_approx"].dropna().values for c in CLASS_NAMES]
    kw_stat, kw_p = stats.kruskal(*groups)
    stat_rows.append({"test": "kruskal_wallis_age_class", "statistic": kw_stat, "p_value": kw_p, "effect_size_cramers_v": float("nan"), "dof": len(CLASS_NAMES) - 1})

    # 4. Point-biserial: age vs malignant
    pb_stat, pb_p = stats.pointbiserialr(df["malignant"], df["age_approx"])
    stat_rows.append({"test": "pointbiserial_age_malignant", "statistic": pb_stat, "p_value": pb_p, "effect_size_cramers_v": abs(pb_stat), "dof": float("nan")})

    stat_df = pd.DataFrame(stat_rows)
    stat_path = tables_dir / "exp09_correlation_statistics.csv"
    stat_df.to_csv(stat_path, index=False)

    # --- Figures ---

    # Age distribution by class
    fig, ax = plt.subplots(figsize=(14, 6))
    order = CLASS_NAMES
    sns.violinplot(data=df, x="class_name", y="age_approx", order=order, ax=ax, palette="Set2")
    ax.set_title("Age Distribution by Diagnosis Class (ISIC 2019)", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Age (years)")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp09_age_distribution_by_class.png", dpi=150)
    plt.close(fig)

    # Site × class heatmap (normalised by site total)
    site_class = pd.crosstab(df["anatom_site_general"], df["class_name"], normalize="index")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(site_class, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Anatomical Site × Class Frequency (row-normalised)", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "exp09_site_class_heatmap.png", dpi=150)
    plt.close(fig)

    # Sex × class barplot
    sex_class = pd.crosstab(df["sex"], df["class_name"], normalize="index")
    sex_class.plot(kind="bar", figsize=(10, 5))
    plt.title("Sex × Class Distribution (row-normalised)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(figures_dir / "exp09_sex_class_barplot.png", dpi=150)
    plt.close()

    print(f"[EXP09] Done. Saved to: {tables_dir} and {figures_dir}")
    print(stat_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
