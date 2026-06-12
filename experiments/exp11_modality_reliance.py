"""Experiment 11 — Per-Class Modality Reliance Map (re-analysis of exp02 CSV)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import chi2_contingency

from data.preprocessing import CLASS_NAMES


def main(cfg: dict) -> None:
    print("[EXP11] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    src = tables_dir / "exp02_occlusion_results.csv"
    df = pd.read_csv(src)

    # exp02 CSV: group = condition label (occlude_none / occlude_image / occlude_metadata)
    # class column holds the diagnostic class name
    rows = []
    for cls in CLASS_NAMES:

        def get_val(grp, col, _cls=cls):
            r = df[(df["group"] == grp) & (df["class"] == _cls)]
            if r.empty:
                return np.nan
            return r.iloc[0][col]

        sens_none = get_val("occlude_none", "sensitivity")
        sens_img = get_val("occlude_image", "sensitivity")
        sens_meta = get_val("occlude_metadata", "sensitivity")
        support = get_val("occlude_none", "support")

        if any(np.isnan(v) for v in [sens_none, sens_img, sens_meta, support]):
            continue

        # Reliance scores (clipped to [0,1])
        rel_image = np.clip((sens_none - sens_img) / (sens_none + 1e-9), 0, 1)
        rel_meta = np.clip((sens_none - sens_meta) / (sens_none + 1e-9), 0, 1)

        # Chi-squared: 2x2 table of TP/FN for image-occ vs meta-occ
        n = int(round(support))
        tp_img = int(round(sens_img * n))
        fn_img = n - tp_img
        tp_meta = int(round(sens_meta * n))
        fn_meta = n - tp_meta

        contingency = np.array([[tp_img, fn_img], [tp_meta, fn_meta]])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
        except Exception:
            chi2, p_val = np.nan, np.nan

        rows.append({
            "class": cls,
            "support": n,
            "sensitivity_none": round(sens_none, 4),
            "sensitivity_img_occ": round(sens_img, 4),
            "sensitivity_meta_occ": round(sens_meta, 4),
            "reliance_image": round(rel_image, 4),
            "reliance_metadata": round(rel_meta, 4),
            "chi2": round(chi2, 4) if not np.isnan(chi2) else np.nan,
            "p_value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        })

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp11_modality_reliance.csv"
    result_df.to_csv(result_path, index=False)

    # Heatmap: classes × [reliance_image, reliance_metadata]
    heat_df = result_df.set_index("class")[["reliance_image", "reliance_metadata"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        heat_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        xticklabels=["Image Reliance", "Metadata Reliance"],
    )
    ax.set_title("Per-Class Modality Reliance (Fusion Model)\n"
                 "Score = sensitivity drop when modality occluded", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Diagnostic Class")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp11_modality_reliance_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n[EXP11] Done. Saved to: {result_path}")
    print(result_df[["class", "reliance_image", "reliance_metadata", "p_value", "significant"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
