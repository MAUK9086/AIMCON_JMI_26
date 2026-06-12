"""Experiment 12c — Fallback pattern comparison across fusion variants under image occlusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]

MODEL_CKPTS = {
    "fusion":             "fusion_best.pth",
    "fusion_dropout":     "fusion_dropout_best.pth",
    "fusion_reweighted":  "fusion_reweighted_best.pth",
}


def infer_occluded(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    mean_image = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="occlude=image"):
            images   = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels   = batch["label"]
            images   = mean_image.expand_as(images)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main(cfg: dict) -> None:
    print("[EXP12C] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    img_size = cfg["data"]["image_size"]
    test_ds  = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    models_dir = Path(cfg["paths"]["models_dir"])

    rows = []
    entropies = {}

    for model_name, ckpt_name in MODEL_CKPTS.items():
        ckpt = models_dir / ckpt_name
        if not ckpt.exists():
            print(f"  [SKIP] {model_name} — checkpoint not found: {ckpt}")
            continue

        # Use fusion batch size for all variants
        bs = cfg["training"].get(f"batch_size_{model_name}", cfg["training"].get("batch_size_fusion", 32))
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        model = build_model("fusion", cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Inferring {model_name} under image occlusion...")

        y_true, y_pred = infer_occluded(model, loader, device)

        # Per-class FNR
        n_classes = len(CLASS_NAMES)
        for i, cls in enumerate(CLASS_NAMES):
            support = int((y_true == i).sum())
            if support == 0:
                continue
            fn  = int(((y_true == i) & (y_pred != i)).sum())
            fnr = fn / support
            rows.append({"model": model_name, "class": cls, "support": support,
                         "fn": fn, "fnr": round(fnr, 4)})

        # Prediction distribution entropy (higher = more diffuse / less anchored)
        pred_counts = np.bincount(y_pred, minlength=n_classes).astype(float)
        pred_dist   = pred_counts / pred_counts.sum()
        ent = scipy_entropy(pred_dist)
        entropies[model_name] = round(ent, 4)
        print(f"    {model_name}: prediction entropy under image occlusion = {ent:.4f}")

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp12c_fallback_comparison.csv"
    result_df.to_csv(result_path, index=False)

    # Bar chart of prediction-distribution entropy per model
    if entropies:
        fig, ax = plt.subplots(figsize=(7, 4))
        models = list(entropies.keys())
        vals   = [entropies[m] for m in models]
        colors = ["#4C72B0", "#DD8452", "#55A868"][:len(models)]
        bars = ax.bar(models, vals, color=colors, width=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.axhline(np.log(len(CLASS_NAMES)), color="gray", linestyle="--", linewidth=1,
                   label=f"Max entropy (uniform) = {np.log(len(CLASS_NAMES)):.2f}")
        ax.set_ylabel("Shannon Entropy of Prediction Distribution")
        ax.set_title("Prediction Distribution Entropy Under Image Occlusion\n"
                     "(higher = more diffuse fallback; lower = anchored to fewer classes)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(figures_dir / "exp12c_fallback_entropy.png", dpi=150)
        plt.close(fig)

    print(f"\n[EXP12C] Done. Saved to: {result_path}")
    pivot = result_df.pivot_table(index="class", columns="model", values="fnr")
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
