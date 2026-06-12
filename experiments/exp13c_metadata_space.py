"""Experiment 13c — Anchor/sink centroid distances + PCA of metadata feature space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import (
    load_and_merge, impute_metadata, stratified_split, encode_metadata, CLASS_NAMES,
)
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, get_batch_size, set_seed


def infer_metadata_only(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="infer"):
            images   = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels   = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main(cfg: dict) -> None:
    print("[EXP13C] Starting...")

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

    # 14-dim encoded metadata for entire test set
    meta_features = encode_metadata(test_df)   # shape (N, 14)

    # Run metadata_only inference to get per-sample predictions
    img_size = cfg["data"]["image_size"]
    test_ds  = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    model_name = "metadata_only"
    bs = get_batch_size(model_name, cfg)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, cfg).to(device)
    ckpt  = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    y_true, y_pred = infer_metadata_only(model, loader, device)

    # Compute centroids for 4 groups of interest
    scc_idx  = CLASS_NAMES.index("SCC")
    nv_idx   = CLASS_NAMES.index("NV")
    ak_idx   = CLASS_NAMES.index("AK")
    df_idx   = CLASS_NAMES.index("DF")

    groups = {
        "true_SCC":         meta_features[y_true == scc_idx],
        "true_NV":          meta_features[y_true == nv_idx],
        "predicted_as_AK":  meta_features[y_pred == ak_idx],
        "predicted_as_DF":  meta_features[y_pred == df_idx],
    }

    centroids = {}
    for grp, feats in groups.items():
        if len(feats) == 0:
            print(f"  [SKIP] {grp} — no samples")
            continue
        centroids[grp] = feats.mean(axis=0)
        print(f"  {grp}: n={len(feats)}")

    # Pairwise distances
    grp_names = list(centroids.keys())
    cent_matrix = np.stack([centroids[g] for g in grp_names])
    dist_matrix = cdist(cent_matrix, cent_matrix, metric="euclidean")
    dist_df = pd.DataFrame(dist_matrix, index=grp_names, columns=grp_names).round(4)
    dist_path = tables_dir / "exp13c_centroid_distances.csv"
    dist_df.to_csv(dist_path)
    print(f"\n  Centroid pairwise distances:")
    print(dist_df.to_string())

    # 2D PCA of full test set metadata features
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(meta_features)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    # Plot scatter (subsample to avoid overplotting)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(coords), size=min(2000, len(coords)), replace=False)
    for i, cls in enumerate(CLASS_NAMES):
        mask = (y_true == i)
        sel  = np.intersect1d(np.where(mask)[0], sample_idx)
        if len(sel) == 0:
            continue
        ax.scatter(coords[sel, 0], coords[sel, 1], c=[cmap(i)],
                   s=10, alpha=0.3, label=cls)

    # Plot centroids for 4 interest groups
    markers = {"true_SCC": "*", "true_NV": "D", "predicted_as_AK": "^", "predicted_as_DF": "s"}
    for grp, centroid in centroids.items():
        c_2d = pca.transform(centroid.reshape(1, -1))[0]
        ax.scatter(c_2d[0], c_2d[1], marker=markers.get(grp, "X"),
                   s=250, edgecolors="black", linewidths=1.5, zorder=5,
                   label=f"Centroid: {grp}", c="white")
        ax.annotate(grp, (c_2d[0], c_2d[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, fontweight="bold")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA of 14-dim Metadata Feature Space (test set)\n"
                 "Scatter = true class; Centroids = anchor/sink groups")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig_path = figures_dir / "exp13c_metadata_pca.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\n[EXP13C] Done. Saved to: {dist_path}, {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
