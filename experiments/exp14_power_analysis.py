"""Experiment 14 — Statistical Power Analysis on Subgroup FNR Gap (Cohen's h)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm

from data.preprocessing import (
    load_and_merge, impute_metadata, stratified_split,
    get_subgroup_mask, CLASS_NAMES,
)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def compute_power(n1: int, n2: int, h: float, alpha: float = 0.05) -> float:
    """Two-sample two-tailed power using Cohen's h."""
    if n1 <= 1 or n2 <= 1 or h == 0:
        return 0.0
    se = np.sqrt(1 / n1 + 1 / n2)
    z = abs(h) / se - norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z))


def n_for_power(target: float, h: float, ratio: float, alpha: float = 0.05,
                max_n: int = 10000) -> int:
    for n1 in range(2, max_n):
        n2 = max(2, int(round(ratio * n1)))
        if compute_power(n1, n2, h, alpha) >= target:
            return n1
    return max_n


def main(cfg: dict) -> None:
    print("[EXP14] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load test set to get positive-class counts per subgroup
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    seed = cfg["project"]["seed"]
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    indian_mask  = get_subgroup_mask(test_df, "indian_profile", cfg)
    western_mask = get_subgroup_mask(test_df, "western_profile", cfg)

    # FNR gap estimates from bootstrap
    src = tables_dir / "exp10_bootstrap_ci.csv"
    boot_df = pd.read_csv(src)
    boot_df = boot_df[boot_df["model"] == "fusion"].copy()

    rows = []
    power_curves = {}

    for _, row in boot_df.iterrows():
        cls = row["class"]
        cls_idx = CLASS_NAMES.index(cls)

        # Positive-class counts WITHIN each subgroup (correct N for FNR power analysis)
        n_indian_pos  = int((indian_mask  & (test_df["label"] == cls_idx)).sum())
        n_western_pos = int((western_mask & (test_df["label"] == cls_idx)).sum())

        p_indian  = float(row["indian_fnr_point"])
        p_western = float(row["western_fnr_point"])

        if n_indian_pos < 5 or n_western_pos < 5:
            rows.append({
                "class": cls,
                "n_indian_pos": n_indian_pos,
                "n_western_pos": n_western_pos,
                "indian_fnr": round(p_indian, 4),
                "western_fnr": round(p_western, 4),
                "cohens_h": np.nan,
                "achieved_power": np.nan,
                "n_required_for_80pct_power": np.nan,
                "note": f"excluded — N too small (indian={n_indian_pos}, western={n_western_pos})",
            })
            continue

        h = cohens_h(p_indian, p_western)
        ratio = n_western_pos / n_indian_pos
        power = compute_power(n_indian_pos, n_western_pos, h)
        n_req = n_for_power(0.80, abs(h), ratio) if abs(h) > 1e-6 else 10000

        # Sanity check: high power + non-significant gap should not coexist
        if power > 0.95 and not row["gap_significant_95pct"]:
            print(f"  WARNING [{cls}]: achieved_power={power:.2f} but gap not significant "
                  f"— check inputs (h={h:.3f}, n_indian={n_indian_pos}, n_western={n_western_pos})")

        rows.append({
            "class": cls,
            "n_indian_pos": n_indian_pos,
            "n_western_pos": n_western_pos,
            "indian_fnr": round(p_indian, 4),
            "western_fnr": round(p_western, 4),
            "cohens_h": round(h, 4),
            "achieved_power": round(power, 4),
            "n_required_for_80pct_power": n_req,
            "note": "",
        })

        ns = np.arange(5, max(n_req * 3, n_indian_pos * 4, 200), 3)
        powers = [compute_power(int(n), max(2, int(round(ratio * n))), abs(h)) for n in ns]
        power_curves[cls] = (ns, powers, n_indian_pos, power, n_req)

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp14_power_analysis.csv"
    result_df.to_csv(result_path, index=False)

    # Power curve figure
    if power_curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"MEL": "#E05C5C", "BCC": "#4C72B0", "SCC": "#55A868"}
        for cls, (ns, powers, n_cur, pwr_cur, n_req) in power_curves.items():
            h_val = result_df[result_df["class"] == cls]["cohens_h"].values[0]
            color = colors.get(cls, "gray")
            ax.plot(ns, powers, color=color, linewidth=2,
                    label=f"{cls}  h={h_val:.2f},  current N={n_cur} (power={pwr_cur:.2f})")
            ax.axvline(n_cur, color=color, linestyle=":", alpha=0.7)
            if n_req < max(ns):
                ax.axvline(n_req, color=color, linestyle="--", alpha=0.5)

        ax.axhline(0.80, color="black", linestyle="--", linewidth=1.2, label="80% power threshold")
        ax.set_xlabel("Subgroup-Positive Sample Size (N_Indian class-positives)")
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1.05)
        ax.set_title("Power Analysis — FNR Gap (Indian − Western)\n"
                     "Fusion model, Cohen's h, α=0.05 two-tailed\n"
                     "(dotted = current N; dashed = N needed for 80% power)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(figures_dir / "exp14_power_curve.png", dpi=150)
        plt.close(fig)

    print(f"\n[EXP14] Done. Saved to: {result_path}")
    print(result_df[["class", "n_indian_pos", "n_western_pos",
                      "cohens_h", "achieved_power",
                      "n_required_for_80pct_power", "note"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
