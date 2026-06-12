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


def _chi2_vs_baseline(tp_none, fn_none, tp_occ, fn_occ):
    """Chi-squared test: does occluding this modality change TP/FN vs baseline?"""
    contingency = np.array([[tp_none, fn_none], [tp_occ, fn_occ]])
    try:
        _, p_val, _, _ = chi2_contingency(contingency, correction=False)
        return float(p_val)
    except Exception:
        return np.nan


def main(cfg: dict) -> None:
    print("[EXP11] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    src = tables_dir / "exp02_occlusion_results.csv"
    df = pd.read_csv(src)

    rows = []
    for cls in CLASS_NAMES:

        def get_val(grp, col, _cls=cls):
            r = df[(df["group"] == grp) & (df["class"] == _cls)]
            if r.empty:
                return np.nan
            return r.iloc[0][col]

        sens_none = get_val("occlude_none", "sensitivity")
        sens_img  = get_val("occlude_image", "sensitivity")
        sens_meta = get_val("occlude_metadata", "sensitivity")
        support   = get_val("occlude_none", "support")

        if any(np.isnan(v) for v in [sens_none, sens_img, sens_meta, support]):
            continue

        rel_image = np.clip((sens_none - sens_img)  / (sens_none + 1e-9), 0, 1)
        rel_meta  = np.clip((sens_none - sens_meta) / (sens_none + 1e-9), 0, 1)

        n = int(round(support))
        tp_none = int(round(sens_none * n)); fn_none = n - tp_none
        tp_img  = int(round(sens_img  * n)); fn_img  = n - tp_img
        tp_meta = int(round(sens_meta * n)); fn_meta = n - tp_meta

        # Two separate tests: does each occlusion significantly change TP/FN vs baseline?
        p_image    = _chi2_vs_baseline(tp_none, fn_none, tp_img,  fn_img)
        p_metadata = _chi2_vs_baseline(tp_none, fn_none, tp_meta, fn_meta)

        rows.append({
            "class": cls,
            "support": n,
            "sensitivity_none":     round(sens_none, 4),
            "sensitivity_img_occ":  round(sens_img,  4),
            "sensitivity_meta_occ": round(sens_meta, 4),
            "reliance_image":    round(rel_image, 4),
            "reliance_metadata": round(rel_meta,  4),
            "p_image":    round(p_image,    4) if not np.isnan(p_image)    else np.nan,
            "p_metadata": round(p_metadata, 4) if not np.isnan(p_metadata) else np.nan,
            "image_significant":    bool(p_image    < 0.05) if not np.isnan(p_image)    else False,
            "metadata_significant": bool(p_metadata < 0.05) if not np.isnan(p_metadata) else False,
        })

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp11_modality_reliance.csv"
    result_df.to_csv(result_path, index=False)

    # Heatmap: classes × [sensitivity_none, reliance_image, reliance_metadata]
    heat_df = result_df.set_index("class")[["sensitivity_none", "reliance_image", "reliance_metadata"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        xticklabels=["Baseline\nSensitivity", "Image\nReliance", "Metadata\nReliance"],
    )

    # Mark significant modality contributions with *
    for i, row in enumerate(result_df.itertuples()):
        if row.image_significant:
            ax.text(1.5, i + 0.5, "*", ha="center", va="center", fontsize=13, color="navy")
        if row.metadata_significant:
            ax.text(2.5, i + 0.5, "*", ha="center", va="center", fontsize=13, color="navy")

    ax.set_title("Per-Class Modality Reliance — Fusion Model\n"
                 "Reliance = sensitivity drop when modality occluded  (* = p<0.05 vs baseline)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Diagnostic Class")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp11_modality_reliance_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n[EXP11] Done. Saved to: {result_path}")
    print(result_df[["class", "reliance_image", "reliance_metadata",
                      "p_image", "image_significant",
                      "p_metadata", "metadata_significant"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
