"""Experiment 8 — Per-Anatomical-Site Performance Analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES, SITE_CATEGORIES
from data.dataset import ISICDataset, get_val_transforms
from evaluation.metrics import balanced_accuracy_score_np
from training.train import build_model, get_batch_size, set_seed


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
    print("[EXP08] Starting...")
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

    malignant_classes = cfg["subgroups"]["malignant_classes"]
    malignant_indices = [CLASS_NAMES.index(c) for c in malignant_classes]
    MEL_IDX = CLASS_NAMES.index("MEL")

    rows = []
    sites = [s for s in SITE_CATEGORIES if s != "unknown"]

    for model_name in ["image_only", "metadata_only", "fusion"]:
        bs = get_batch_size(model_name, cfg)
        loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(model_name, cfg).to(device)
        ckpt = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        y_true, y_pred, y_prob = infer(model, loader, device)

        for site in sites:
            mask = (test_df["anatom_site_general"] == site).values
            n = mask.sum()
            if n < 3:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            # MEL sensitivity
            mel_mask = (yt == MEL_IDX)
            mel_n = mel_mask.sum()
            mel_sens = float((yp[mel_mask] == MEL_IDX).sum() / (mel_n + 1e-9)) if mel_n > 0 else float("nan")

            # Per malignant class sensitivity
            for cls, cidx in zip(malignant_classes, malignant_indices):
                cls_mask = (yt == cidx)
                cls_n = cls_mask.sum()
                sens = float((yp[cls_mask] == cidx).sum() / (cls_n + 1e-9)) if cls_n > 0 else float("nan")
                rows.append({
                    "model": model_name,
                    "site": site,
                    "class": cls,
                    "n_total": int(n),
                    "n_class": int(cls_n),
                    "sensitivity": sens,
                    "low_n_flag": cls_n < 10,
                })

    result_df = pd.DataFrame(rows)
    result_path = tables_dir / "exp08_site_stratified.csv"
    result_df.to_csv(result_path, index=False)

    # Heatmap: model × site × MEL sensitivity
    mel_df = result_df[result_df["class"] == "MEL"]
    pivot = mel_df.pivot_table(index="model", columns="site", values="sensitivity")

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5), 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
    ax.set_title("MEL Sensitivity by Anatomical Site — All Models\n(flag: sites with N<10 are unreliable)")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp08_site_sensitivity_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[EXP08] Done. Saved to: {result_path}")
    print(mel_df[["model", "site", "n_class", "sensitivity", "low_n_flag"]].sort_values(["site", "model"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
