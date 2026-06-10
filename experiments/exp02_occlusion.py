"""Experiment 2 — Modality Occlusion Test on trained fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from evaluation.metrics import compute_full_metrics, balanced_accuracy_score_np
from training.train import build_model, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]


def infer_occluded(model, loader, device, occlude: str) -> tuple:
    """occlude: 'none' | 'image' | 'metadata'"""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    mean_image = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"occlude={occlude}"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]

            if occlude == "image":
                images = mean_image.expand_as(images)
            elif occlude == "metadata":
                metadata = torch.zeros_like(metadata)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.append(labels.numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def main(cfg: dict) -> None:
    print("[EXP02] Starting...")
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

    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(cfg["data"]["image_size"]))
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size_fusion"],
                             shuffle=False, num_workers=4, pin_memory=True)

    model = build_model("fusion", cfg).to(device)
    ckpt = Path(cfg["paths"]["models_dir"]) / "fusion_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    results = []
    for occlude in ["none", "image", "metadata"]:
        y_true, y_pred, y_prob = infer_occluded(model, test_loader, device, occlude)
        metrics_df = compute_full_metrics(y_true, y_pred, y_prob, label=f"occlude_{occlude}")
        metrics_df["condition"] = occlude
        results.append(metrics_df)

    result_df = pd.concat(results, ignore_index=True)
    result_path = tables_dir / "exp02_occlusion_results.csv"
    result_df.to_csv(result_path, index=False)

    # Bar chart: balanced accuracy drop per class by occlusion condition
    pivot = result_df[result_df["class"] != "AGGREGATE"].pivot_table(
        index="class", columns="condition", values="sensitivity"
    )
    pivot = pivot.reindex(index=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    for i, cond in enumerate(["none", "image", "metadata"]):
        ax.bar(x + i * w, pivot[cond].values, width=w, label=f"Occlude {cond}")
    ax.set_xticks(x + w)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Sensitivity (Recall)")
    ax.set_title("Modality Occlusion Test — Per-Class Sensitivity")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "exp02_occlusion_barplot.png", dpi=150)
    plt.close(fig)

    print(f"[EXP02] Done. Saved to: {result_path}")
    agg = result_df[result_df["class"] == "AGGREGATE"][["condition", "sensitivity", "auc"]]
    print(agg.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
