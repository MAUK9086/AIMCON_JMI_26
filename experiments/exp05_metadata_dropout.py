"""Experiment 5 — Metadata Dropout Mitigation: retrain fusion with p=0.5 metadata dropout."""

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

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES, get_subgroup_mask
from data.dataset import ISICDataset, get_val_transforms
from evaluation.metrics import compute_full_metrics
from evaluation.subgroup_eval import evaluate_subgroups
from training.train import train_model, build_model, get_batch_size, set_seed


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
    print("[EXP05] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dropout_prob = cfg["experiments"]["metadata_dropout_prob"]

    # Retrain fusion with metadata dropout
    print(f"[EXP05] Training fusion_dropout with metadata_dropout_prob={dropout_prob}")
    train_model("fusion_dropout", cfg, metadata_dropout_prob=dropout_prob)

    # Evaluate on test set
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    img_size = cfg["data"]["image_size"]
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    bs = cfg["training"]["batch_size_fusion"]
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    all_results = []
    for model_name in ["fusion", "fusion_dropout"]:
        model = build_model(model_name, cfg).to(device)
        ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        y_true, y_pred, y_prob = infer(model, loader, device)
        subgroup_df = evaluate_subgroups(test_df, y_true, y_pred, y_prob, cfg, model_name)
        all_results.append(subgroup_df)

    result_df = pd.concat(all_results, ignore_index=True)
    result_path = tables_dir / "exp05_dropout_comparison.csv"
    result_df.to_csv(result_path, index=False)

    # Plot: before/after FNR for malignant classes by subgroup
    malignant = cfg["subgroups"]["malignant_classes"]
    fnr_rows = result_df[
        result_df["class"].isin(malignant) &
        result_df["group"].str.contains("profile")
    ].copy()
    fnr_rows["profile"] = fnr_rows["group"].str.split("/").str[-1]

    fig, axes = plt.subplots(1, len(malignant), figsize=(5 * len(malignant), 5))
    import seaborn as sns
    for ax, cls in zip(axes, malignant):
        sub = fnr_rows[fnr_rows["class"] == cls]
        sns.barplot(data=sub, x="profile", y="fnr", hue="model", ax=ax)
        ax.set_title(f"FNR — {cls}")
        ax.set_ylim(0, 1)
    plt.suptitle("Mitigation Effect: Metadata Dropout (fusion vs fusion_dropout)")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp05_mitigation_effect.png", dpi=150)
    plt.close(fig)

    print(f"[EXP05] Done. Saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
