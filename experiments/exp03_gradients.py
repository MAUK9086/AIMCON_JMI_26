"""Experiment 3 — Integrated Gradients attribution on fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, set_seed

METADATA_FEATURE_NAMES = [
    "age_norm",
    "sex_male", "sex_female", "sex_unknown",
    "site_anterior_torso", "site_head_neck", "site_lateral_torso",
    "site_lower_extremity", "site_oral_genital", "site_palms_soles",
    "site_posterior_torso", "site_upper_extremity", "site_unknown",
    "site_unknown2",
]

TARGET_CLASSES = {"MEL": 0, "NV": 1, "BCC": 2, "BKL": 4}
SAMPLES_PER_CLASS = 50


class FusionWrapper(torch.nn.Module):
    """Wraps fusion model to accept (image, metadata) as a single concatenated input for captum."""
    def __init__(self, model, image_shape):
        super().__init__()
        self.model = model
        self.image_numel = int(np.prod(image_shape))

    def forward(self, x):
        image = x[:, :self.image_numel].view(x.shape[0], *[3, 224, 224])
        metadata = x[:, self.image_numel:]
        return self.model(image=image, metadata=metadata)


def main(cfg: dict) -> None:
    print("[EXP03] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    gradcam_dir = figures_dir / "exp03_gradcam_samples"
    for d in [tables_dir, figures_dir, gradcam_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(cfg["data"]["image_size"]))

    model = build_model("fusion", cfg).to(device)
    ckpt = Path(cfg["paths"]["models_dir"]) / "fusion_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Collect 50 samples per target class
    selected_indices = []
    rng = np.random.default_rng(seed)
    for cls_name, cls_idx in TARGET_CLASSES.items():
        class_indices = np.where(test_df["label"].values == cls_idx)[0]
        n = min(SAMPLES_PER_CLASS, len(class_indices))
        chosen = rng.choice(class_indices, n, replace=False)
        selected_indices.extend(chosen.tolist())

    subset_ds = Subset(test_ds, selected_indices)
    loader = DataLoader(subset_ds, batch_size=16, shuffle=False, num_workers=2)

    # --- Integrated Gradients on metadata features ---
    try:
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(model)
    except ImportError:
        print("[EXP03] captum not installed — skipping IG attribution.")
        ig = None

    meta_attribs = {cls: np.zeros(14) for cls in TARGET_CLASSES}
    meta_counts = {cls: 0 for cls in TARGET_CLASSES}

    if ig is not None:
        idx_ptr = 0
        for batch in tqdm(loader, desc="IG attribution"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]

            for i in range(len(labels)):
                lbl = labels[i].item()
                cls_name = {v: k for k, v in TARGET_CLASSES.items()}.get(lbl)
                if cls_name is None:
                    continue
                img_i = images[i:i+1].requires_grad_(True)
                meta_i = metadata[i:i+1].requires_grad_(True)

                try:
                    attr_img, attr_meta = ig.attribute(
                        inputs=(img_i, meta_i),
                        target=lbl,
                        n_steps=25,
                        internal_batch_size=1,
                    )
                    meta_attribs[cls_name] += attr_meta.abs().detach().cpu().numpy()[0]
                    meta_counts[cls_name] += 1
                except Exception as e:
                    pass

    # Average attributions
    attr_matrix = np.zeros((len(TARGET_CLASSES), 14))
    for i, cls_name in enumerate(TARGET_CLASSES):
        if meta_counts[cls_name] > 0:
            attr_matrix[i] = meta_attribs[cls_name] / meta_counts[cls_name]

    feat_names = METADATA_FEATURE_NAMES[:14]
    attr_df = pd.DataFrame(attr_matrix, index=list(TARGET_CLASSES.keys()), columns=feat_names)

    feat_importance = attr_df.T
    feat_importance.index.name = "feature"
    feat_importance_path = tables_dir / "exp03_feature_importance.csv"
    feat_importance.reset_index().to_csv(feat_importance_path, index=False)

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(attr_df, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax)
    ax.set_title("Integrated Gradients: Mean |Attribution| per Metadata Feature (Fusion Model)")
    ax.set_xlabel("Metadata Feature")
    ax.set_ylabel("Class")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp03_metadata_attributions.png", dpi=150)
    plt.close(fig)

    # --- GradCAM on image branch ---
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import cv2

        target_layer = model.image_encoder.blocks[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])

        saved = 0
        for batch in tqdm(loader, desc="GradCAM"):
            if saved >= 20:
                break
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"]

            for i in range(min(len(labels), 20 - saved)):
                img_t = images[i:i+1]
                meta_t = metadata[i:i+1]

                class FusionInputWrapper(torch.nn.Module):
                    def __init__(self, m, meta):
                        super().__init__()
                        self.m = m
                        self.meta = meta
                    def forward(self, x):
                        return self.m(image=x, metadata=self.meta)

                wrapper = FusionInputWrapper(model, meta_t)
                try:
                    cam_obj = GradCAM(model=wrapper, target_layers=[model.image_encoder.blocks[-1]])
                    grayscale_cam = cam_obj(input_tensor=img_t)[0]
                    img_np = img_t[0].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-9)
                    cam_image = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
                    plt.imsave(gradcam_dir / f"gradcam_{saved:02d}_class{labels[i].item()}.png", cam_image)
                    saved += 1
                except Exception:
                    pass
    except ImportError:
        print("[EXP03] grad-cam not installed — skipping GradCAM.")

    print(f"[EXP03] Done. Saved to: {figures_dir} and {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
