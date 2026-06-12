"""Experiment 14 — Statistical Power Analysis on Subgroup FNR Gap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm


def compute_power(n1: int, n2: int, effect_size: float, alpha: float = 0.05) -> float:
    """Two-sample two-tailed z-test power."""
    if n1 <= 0 or n2 <= 0 or effect_size == 0:
        return 0.0
    se = np.sqrt(1 / n1 + 1 / n2)
    z_alpha = norm.ppf(1 - alpha / 2)
    z = abs(effect_size) / se - z_alpha
    return float(norm.cdf(z))


def n_for_power(target_power: float, effect_size: float, ratio: float = 1.0,
                alpha: float = 0.05, max_n: int = 50000) -> int:
    """Find minimum n1 such that power >= target_power (n2 = ratio * n1)."""
    for n1 in range(2, max_n):
        n2 = max(2, int(round(ratio * n1)))
        if compute_power(n1, n2, effect_size, alpha) >= target_power:
            return n1
    return max_n


def main(cfg: dict) -> None:
    print("[EXP14] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    src = tables_dir / "exp10_bootstrap_ci.csv"
    df = pd.read_csv(src)

    # Use fusion model rows only
    df = df[df["model"] == "fusion"].copy()

    rows = []
    power_curves = {}

    for _, row in df.iterrows():
        cls = row["class"]
        n_indian = int(row["n_indian"])
        n_western = int(row["n_western"])
        gap = float(row["gap_point"])
        ci_low = float(row["gap_ci_low"])
        ci_high = float(row["gap_ci_high"])

        # Estimate pooled std from 95% CI width
        ci_width = ci_high - ci_low
        pooled_std = ci_width / (2 * 1.96) if ci_width > 0 else np.nan

        if np.isnan(pooled_std) or pooled_std == 0 or gap == 0:
            rows.append({
                "class": cls,
                "n_indian": n_indian,
                "n_western": n_western,
                "observed_gap": round(gap, 4),
                "pooled_std_est": np.nan,
                "effect_size": np.nan,
                "achieved_power": np.nan,
                "n_required_for_80pct_power": np.nan,
                "note": "insufficient gap or CI width for power estimate",
            })
            continue

        effect_size = abs(gap) / pooled_std
        achieved_power = compute_power(n_indian, n_western, effect_size)
        ratio = n_western / n_indian if n_indian > 0 else 1.0
        n_req = n_for_power(0.80, effect_size, ratio=ratio)

        rows.append({
            "class": cls,
            "n_indian": n_indian,
            "n_western": n_western,
            "observed_gap": round(gap, 4),
            "pooled_std_est": round(pooled_std, 4),
            "effect_size": round(effect_size, 4),
            "achieved_power": round(achieved_power, 4),
            "n_required_for_80pct_power": n_req,
            "note": "",
        })

        # Power curve data for plotting
        ns = np.arange(10, max(n_req * 2, n_indian * 3, 500), 5)
        powers = [compute_power(int(n), max(2, int(round(ratio * n))), effect_size) for n in ns]
        power_curves[cls] = (ns, powers, n_indian, achieved_power, n_req)

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp14_power_analysis.csv"
    result_df.to_csv(result_path, index=False)

    # Power curve figure
    if power_curves:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"MEL": "#E05C5C", "BCC": "#4C72B0", "SCC": "#55A868"}
        for cls, (ns, powers, n_current, pwr_current, n_req) in power_curves.items():
            color = colors.get(cls, "gray")
            ax.plot(ns, powers, label=f"{cls} (d={result_df[result_df['class']==cls]['effect_size'].values[0]:.2f})",
                    color=color, linewidth=2)
            ax.axvline(n_current, color=color, linestyle=":", alpha=0.7,
                       label=f"{cls} current N={n_current} (power={pwr_current:.2f})")

        ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="80% power threshold")
        ax.set_xlabel("Sample Size per Subgroup (N_Indian)")
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1.05)
        ax.set_title("Power Analysis — FNR Gap (Indian − Western)\nFusion Model, α=0.05 two-tailed")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(figures_dir / "exp14_power_curve.png", dpi=150)
        plt.close(fig)

    print(f"\n[EXP14] Done. Saved to: {result_path}")
    print(result_df[["class", "n_indian", "effect_size", "achieved_power", "n_required_for_80pct_power"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
