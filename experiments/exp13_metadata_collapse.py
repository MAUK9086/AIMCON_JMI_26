"""Experiment 13 — Metadata-Only Model Collapse Analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, get_batch_size, set_seed

COLLAPSE_THRESHOLD = 0.30


def infer(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="infer"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main(cfg: dict) -> None:
    print("[EXP13] Starting...")
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
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    model_name = "metadata_only"
    bs = get_batch_size(model_name, cfg)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, cfg).to(device)
    ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    print("  Running metadata-only inference...")
    y_true, y_pred = infer(model, loader, device)

    n_classes = len(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = tables_dir / "exp13_metadata_collapse_matrix.csv"
    cm_df.to_csv(cm_path)

    # Identify collapse pairs: true_class where >THRESHOLD of samples go to predicted_class
    collapse_rows = []
    for i, true_cls in enumerate(CLASS_NAMES):
        support = int((y_true == i).sum())
        if support == 0:
            continue
        for j, pred_cls in enumerate(CLASS_NAMES):
            if i == j:
                continue
            count = int(cm[i, j])
            rate = count / support
            if rate >= COLLAPSE_THRESHOLD:
                collapse_rows.append({
                    "true_class": true_cls,
                    "predicted_class": pred_cls,
                    "count": count,
                    "support": support,
                    "collapse_rate": round(rate, 4),
                })

    collapse_df = pd.DataFrame(collapse_rows).sort_values("collapse_rate", ascending=False)
    collapse_path = tables_dir / "exp13_collapse_pairs.csv"
    collapse_df.to_csv(collapse_path, index=False)

    # Normalized confusion heatmap with collapse annotations
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)

    # Highlight collapse cells with a border
    for _, row in collapse_df.iterrows():
        i = CLASS_NAMES.index(row["true_class"])
        j = CLASS_NAMES.index(row["predicted_class"])
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))

    ax.set_title(f"Metadata-Only Model — Confusion Matrix\n"
                 f"(red border = collapse ≥{int(COLLAPSE_THRESHOLD*100)}% of true class)", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp13_metadata_collapse_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n[EXP13] Done. Saved to: {cm_path}")
    if not collapse_df.empty:
        print(f"  Collapse pairs (≥{int(COLLAPSE_THRESHOLD*100)}%):")
        print(collapse_df.to_string(index=False))
    else:
        print("  No collapse pairs found above threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
