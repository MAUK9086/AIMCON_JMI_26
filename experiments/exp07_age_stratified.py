"""Experiment 7 — Age-Decade Stratified Analysis."""

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
from evaluation.metrics import balanced_accuracy_score_np, compute_full_metrics
from training.train import build_model, get_batch_size, set_seed

AGE_BINS = [0, 30, 40, 50, 60, 70, 80, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "≥80"]


def infer(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def main(cfg: dict) -> None:
    print("[EXP07] Starting...")
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

    test_df = test_df.copy()
    test_df["age_decade"] = pd.cut(test_df["age_approx"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    img_size = cfg["data"]["image_size"]
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    rows = []
    MEL_IDX = CLASS_NAMES.index("MEL")

    for model_name in ["image_only", "metadata_only", "fusion"]:
        bs = get_batch_size(model_name, cfg)
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(model_name, cfg).to(device)
        ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        y_true, y_pred, y_prob = infer(model, loader, device)

        for decade in AGE_LABELS:
            mask = (test_df["age_decade"] == decade).values
            if mask.sum() < 5:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            ypr = y_prob[mask]
            bal_acc = balanced_accuracy_score_np(yt, yp)

            # MEL sensitivity
            mel_mask = (yt == MEL_IDX)
            mel_sens = (yp[mel_mask] == MEL_IDX).sum() / (mel_mask.sum() + 1e-9) if mel_mask.sum() > 0 else float("nan")

            rows.append({
                "model": model_name,
                "age_decade": decade,
                "n": int(mask.sum()),
                "balanced_accuracy": bal_acc,
                "mel_sensitivity": mel_sens,
            })

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp07_age_stratified.csv"
    result_df.to_csv(result_path, index=False)

    # Line plot: MEL sensitivity vs age decade per model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["balanced_accuracy", "mel_sensitivity"]):
        for model_name in ["image_only", "metadata_only", "fusion"]:
            sub = result_df[result_df["model"] == model_name]
            ax.plot(sub["age_decade"], sub[metric], marker="o", label=model_name)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Age Decade")
        ax.set_xlabel("Age Decade")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Age-Stratified Model Performance (ISIC 2019 Test Set)")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp07_age_performance_curve.png", dpi=150)
    plt.close(fig)

    print(f"[EXP07] Done. Saved to: {result_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
