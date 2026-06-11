"""Experiment 10 — Bootstrap Confidence Intervals on Subgroup FNR Gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import (
    load_and_merge, impute_metadata, stratified_split,
    CLASS_NAMES, get_subgroup_mask
)
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, get_batch_size, set_seed

N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
FOCUS_CLASSES = ["MEL", "BCC", "SCC"]


def infer(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def bootstrap_fnr(y_true, y_pred, class_idx, n_bootstrap, rng):
    """Bootstrap FNR for a single class. Returns array of length n_bootstrap."""
    n = len(y_true)
    fnr_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        pos_mask = (yt == class_idx)
        if pos_mask.sum() == 0:
            continue
        fn = ((yt == class_idx) & (yp != class_idx)).sum()
        tp = ((yt == class_idx) & (yp == class_idx)).sum()
        fnr_samples.append(fn / (fn + tp + 1e-9))
    return np.array(fnr_samples)


def ci(samples, confidence=0.95):
    alpha = (1 - confidence) / 2
    return np.percentile(samples, [alpha * 100, (1 - alpha) * 100])


def main(cfg: dict) -> None:
    print("[EXP10] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    indian_mask = get_subgroup_mask(test_df, "indian_profile", cfg)
    western_mask = get_subgroup_mask(test_df, "western_profile", cfg)

    print(f"  Indian-profile N={indian_mask.sum()}  Western-profile N={western_mask.sum()}")

    img_size = cfg["data"]["image_size"]
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    rows = []
    plot_data = {}  # model -> class -> {indian_ci, western_ci, gap_ci}

    for model_name in ["image_only", "fusion"]:
        bs = get_batch_size(model_name, cfg)
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        model = build_model(model_name, cfg).to(device)
        ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        print(f"\n  Bootstrapping {model_name} ({N_BOOTSTRAP} iterations)...")
        y_true, y_pred = infer(model, loader, device)

        plot_data[model_name] = {}

        for cls in FOCUS_CLASSES:
            cls_idx = CLASS_NAMES.index(cls)

            # Per-subgroup bootstrap
            indian_fnr = bootstrap_fnr(
                y_true[indian_mask], y_pred[indian_mask], cls_idx, N_BOOTSTRAP, rng
            )
            western_fnr = bootstrap_fnr(
                y_true[western_mask], y_pred[western_mask], cls_idx, N_BOOTSTRAP, rng
            )

            if len(indian_fnr) < 10 or len(western_fnr) < 10:
                print(f"    [SKIP] {cls} — insufficient positive samples in one group")
                continue

            # Gap bootstrap (paired)
            n_paired = min(len(indian_fnr), len(western_fnr))
            gap_samples = indian_fnr[:n_paired] - western_fnr[:n_paired]

            i_point = indian_fnr.mean()
            w_point = western_fnr.mean()
            gap_point = i_point - w_point

            i_ci = ci(indian_fnr)
            w_ci = ci(western_fnr)
            gap_ci = ci(gap_samples)

            # Is gap significant? (CI excludes 0)
            significant = gap_ci[0] > 0

            print(f"    {cls}: Indian FNR={i_point:.3f} [{i_ci[0]:.3f},{i_ci[1]:.3f}]  "
                  f"Western={w_point:.3f} [{w_ci[0]:.3f},{w_ci[1]:.3f}]  "
                  f"Gap={gap_point:.3f} [{gap_ci[0]:.3f},{gap_ci[1]:.3f}]"
                  f"{'  *** SIGNIFICANT' if significant else ''}")

            plot_data[model_name][cls] = {
                "indian_point": i_point, "indian_ci": i_ci,
                "western_point": w_point, "western_ci": w_ci,
                "gap_point": gap_point, "gap_ci": gap_ci,
                "significant": significant,
            }

            rows.append({
                "model": model_name,
                "class": cls,
                "indian_fnr_point": round(i_point, 4),
                "indian_fnr_ci_low": round(i_ci[0], 4),
                "indian_fnr_ci_high": round(i_ci[1], 4),
                "western_fnr_point": round(w_point, 4),
                "western_fnr_ci_low": round(w_ci[0], 4),
                "western_fnr_ci_high": round(w_ci[1], 4),
                "gap_point": round(gap_point, 4),
                "gap_ci_low": round(gap_ci[0], 4),
                "gap_ci_high": round(gap_ci[1], 4),
                "gap_significant_95pct": significant,
                "n_indian": int(indian_mask.sum()),
                "n_western": int(western_mask.sum()),
            })

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp10_bootstrap_ci.csv"
    result_df.to_csv(result_path, index=False)

    # ── Figure: FNR with 95% CI error bars, Indian vs Western, per class per model ──
    focus_cls = [c for c in FOCUS_CLASSES if any(c in plot_data[m] for m in plot_data)]
    n_cls = len(focus_cls)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 5), sharey=True)
    if n_cls == 1:
        axes = [axes]

    colors = {"indian_profile": "#E05C5C", "western_profile": "#4C72B0"}
    model_markers = {"image_only": "o", "fusion": "s"}
    x_positions = {"image_only": 0, "fusion": 1}
    x_labels = ["image_only", "fusion"]

    for ax, cls in zip(axes, focus_cls):
        for profile, color in colors.items():
            for model_name in ["image_only", "fusion"]:
                if cls not in plot_data.get(model_name, {}):
                    continue
                d = plot_data[model_name][cls]
                x = x_positions[model_name] + (0.15 if profile == "western_profile" else -0.15)
                point = d["indian_point"] if profile == "indian_profile" else d["western_point"]
                lo, hi = d["indian_ci"] if profile == "indian_profile" else d["western_ci"]
                ax.errorbar(x, point, yerr=[[point - lo], [hi - point]],
                            fmt=model_markers[model_name], color=color,
                            capsize=5, capthick=1.5, markersize=8, linewidth=1.5)

                # Mark significant gaps
                if profile == "indian_profile" and d.get("significant"):
                    ax.text(x_positions[model_name], max(d["indian_point"], d["western_point"]) + 0.04,
                            "*", ha="center", fontsize=14, color="black")

        ax.set_title(f"{cls} False Negative Rate", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("FNR (lower = better)" if cls == focus_cls[0] else "")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    indian_patch = mpatches.Patch(color=colors["indian_profile"], label="Indian-profile")
    western_patch = mpatches.Patch(color=colors["western_profile"], label="Western-profile")
    fig.legend(handles=[indian_patch, western_patch], loc="upper right", fontsize=10)
    fig.suptitle(
        f"FNR with 95% Bootstrap CI — Indian vs Western Profile\n"
        f"(N_indian={indian_mask.sum()}, N_western={western_mask.sum()}, "
        f"* = gap CI excludes 0)",
        fontsize=11
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "exp10_bootstrap_fnr_ci.png", dpi=150)
    plt.close(fig)

    # ── Figure: Gap plot (Indian FNR − Western FNR) with CI ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    y_pos = 0
    yticks, ylabels = [], []
    for model_name in ["fusion", "image_only"]:
        for cls in focus_cls:
            if cls not in plot_data.get(model_name, {}):
                continue
            d = plot_data[model_name][cls]
            gap = d["gap_point"]
            lo, hi = d["gap_ci"]
            color = "#E05C5C" if d["significant"] else "#888888"
            ax2.barh(y_pos, gap, xerr=[[gap - lo], [hi - gap]],
                     color=color, alpha=0.8, capsize=4, height=0.5)
            yticks.append(y_pos)
            ylabels.append(f"{model_name}\n{cls}")
            y_pos += 1
        y_pos += 0.5  # gap between models

    ax2.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=9)
    ax2.set_xlabel("FNR Gap (Indian − Western)\nPositive = Indian-profile has higher FNR")
    ax2.set_title("FNR Gap with 95% Bootstrap CI\n(red = statistically significant gap)")
    plt.tight_layout()
    fig2.savefig(figures_dir / "exp10_fnr_gap_ci.png", dpi=150)
    plt.close(fig2)

    print(f"\n[EXP10] Done. Saved to: {result_path}")
    print(result_df[["model", "class", "gap_point", "gap_ci_low", "gap_ci_high", "gap_significant_95pct"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
