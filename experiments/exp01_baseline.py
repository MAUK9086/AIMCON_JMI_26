"""Experiment 1 — Train all 3 baseline models and evaluate on test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from evaluation.metrics import compute_full_metrics
from training.train import build_model, get_batch_size, set_seed, train_model


def infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="infer"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.append(labels.numpy())
            all_preds.append(preds)
            all_probs.append(probs)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def main(cfg: dict) -> None:
    print("[EXP01] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    for d in [tables_dir, figures_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Train all 3 models (skip if checkpoint already exists)
    for model_name in ["image_only", "metadata_only", "fusion"]:
        ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
        if ckpt.exists():
            print(f"\n  [SKIP] {model_name} checkpoint already exists: {ckpt}")
            continue
        print(f"\n{'='*50}")
        print(f"  Training: {model_name}")
        print(f"{'='*50}")
        train_model(model_name, cfg)

    # Load test set
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    img_size = cfg["data"]["image_size"]
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    all_metrics = []
    all_cm_data = {}
    all_roc_data = {}

    for model_name in ["image_only", "metadata_only", "fusion"]:
        bs = get_batch_size(model_name, cfg)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(model_name, cfg).to(device)
        ckpt = models_dir / f"{model_name}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        y_true, y_pred, y_prob = infer(model, test_loader, device)
        metrics_df = compute_full_metrics(y_true, y_pred, y_prob, label=model_name)
        metrics_df["model"] = model_name
        all_metrics.append(metrics_df)

        all_cm_data[model_name] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        all_roc_data[model_name] = (y_true, y_prob)

    # Save metrics
    result_df = pd.concat(all_metrics, ignore_index=True)
    result_path = tables_dir / "exp01_baseline_metrics.csv"
    result_df.to_csv(result_path, index=False)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (name, cm) in zip(axes, all_cm_data.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, ax=ax, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp01_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ROC curves (macro OvR per model)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue", "orange", "green"]
    for (model_name, (y_true, y_prob)), color in zip(all_roc_data.items(), colors):
        n_classes = len(CLASS_NAMES)
        fpr_all, tpr_all = [], []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            fpr_all.append(fpr)
            tpr_all.append(tpr)
        mean_auc = np.mean([auc(f, t) for f, t in zip(fpr_all, tpr_all)])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        for i in range(n_classes):
            ax.plot(fpr_all[i], tpr_all[i], alpha=0.2, color=color)
        ax.plot([], [], color=color, label=f"{model_name} (macro AUC={mean_auc:.3f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "exp01_roc_curves.png", dpi=150)
    plt.close(fig)

    print(f"\n[EXP01] Done. Saved to: {result_path}")
    summary = result_df[result_df["class"] == "AGGREGATE"][["model", "sensitivity", "f1", "auc"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
