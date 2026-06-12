"""Experiment 12 — Full 8×8 Confusion Matrix Under Image Occlusion (fusion model)."""

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]


def infer_occluded(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    mean_image = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="occlude=image"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]
            images = mean_image.expand_as(images)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def main(cfg: dict) -> None:
    print("[EXP12] Starting...")
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

    model_name = "fusion"
    bs = get_batch_size(model_name, cfg)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, cfg).to(device)
    ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    print("  Running inference with image occluded...")
    y_true, y_pred = infer_occluded(model, loader, device)

    n_classes = len(CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = tables_dir / "exp12_occlusion_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    # Per-class FNR table
    fnr_rows = []
    for i, cls in enumerate(CLASS_NAMES):
        support = int((y_true == i).sum())
        tp = int(cm[i, i])
        fn = support - tp
        fnr = fn / (support + 1e-9) if support > 0 else np.nan
        fnr_rows.append({"class": cls, "support": support, "tp": tp, "fn": fn, "fnr": round(fnr, 4)})
    fnr_df = pd.DataFrame(fnr_rows)
    fnr_path = tables_dir / "exp12_per_class_fnr.csv"
    fnr_df.to_csv(fnr_path, index=False)

    # Normalized confusion heatmap
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Fusion Model — Confusion Matrix Under Image Occlusion\n"
                 "(image replaced with ImageNet mean; metadata only)", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp12_occlusion_confusion_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\n[EXP12] Done. Saved to: {cm_path}")
    print(fnr_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
