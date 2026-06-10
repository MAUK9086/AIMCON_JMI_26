"""Core training loop — reusable across all model types."""

from __future__ import annotations

import argparse
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from data.preprocessing import (
    load_and_merge, impute_metadata, stratified_split, compute_class_weights, CLASS_NAMES
)
from data.dataset import ISICDataset, get_train_transforms, get_val_transforms
from training.losses import build_weighted_ce_loss
from training.callbacks import EarlyStopping
from evaluation.metrics import balanced_accuracy_score_np


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, cfg: dict):
    num_classes = cfg["models"]["num_classes"]
    dropout = cfg["models"]["dropout"]

    if model_name == "image_only":
        from models.image_only import ImageOnlyModel
        return ImageOnlyModel(num_classes=num_classes, pretrained=cfg["models"]["image_pretrained"], dropout=dropout)
    elif model_name == "metadata_only":
        from models.metadata_only import MetadataOnlyModel
        return MetadataOnlyModel(
            input_dim=cfg["metadata_features"]["total_dim"],
            hidden_dims=cfg["models"]["metadata_hidden"],
            feature_dim=cfg["models"]["metadata_feature_dim"],
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_name in ("fusion", "fusion_dropout", "fusion_reweighted"):
        from models.fusion import FusionModel
        return FusionModel(
            num_classes=num_classes,
            metadata_input_dim=cfg["metadata_features"]["total_dim"],
            metadata_hidden_dims=cfg["models"]["metadata_hidden"],
            metadata_feature_dim=cfg["models"]["metadata_feature_dim"],
            fusion_hidden_dims=cfg["models"]["fusion_hidden"],
            dropout=dropout,
            image_pretrained=cfg["models"]["image_pretrained"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_batch_size(model_name: str, cfg: dict) -> int:
    if model_name == "metadata_only":
        return cfg["training"]["batch_size_metadata"]
    elif "fusion" in model_name:
        return cfg["training"]["batch_size_fusion"]
    return cfg["training"]["batch_size_image"]


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else torch.no_grad().__class__()

    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    bal_acc = balanced_accuracy_score_np(all_labels, all_preds)
    return avg_loss, bal_acc


def train_model(
    model_name: str,
    cfg: dict,
    metadata_dropout_prob: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
) -> str:
    """Train a model and return path to saved checkpoint."""
    seed = cfg["project"]["seed"]
    set_seed(seed)

    device_str = cfg["project"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Ensure PyTorch 2.7.0+cu128 is installed "
            "and NVIDIA RTX Pro 4000 Blackwell drivers are loaded."
        )
    device = torch.device(device_str)

    # Data
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    train_df, val_df, test_df = stratified_split(
        df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed
    )

    img_size = cfg["data"]["image_size"]
    train_ds = ISICDataset(train_df, cfg["paths"]["images_dir"], get_train_transforms(img_size), metadata_dropout_prob)
    val_ds = ISICDataset(val_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    bs = get_batch_size(model_name, cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_name, cfg).to(device)

    # Class weights
    class_weights = compute_class_weights(train_df["label"].values, cfg["models"]["num_classes"])
    criterion = build_weighted_ce_loss(class_weights, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.amp.GradScaler()

    save_path = str(Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth")
    stopper = EarlyStopping(patience=cfg["training"]["early_stopping_patience"], mode="max", save_path=save_path)

    print(f"[TRAIN] Starting {model_name} on {device}")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_bal = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_bal = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)
        scheduler.step()

        improved = stopper.step(val_bal, model)
        marker = " *" if improved else ""
        print(f"  Epoch {epoch+1:02d} | train_loss={train_loss:.4f} train_bal={train_bal:.4f} | "
              f"val_loss={val_loss:.4f} val_bal={val_bal:.4f}{marker}")

        if stopper.should_stop:
            print(f"  Early stopping at epoch {epoch+1}. Best val_bal={stopper.best_score:.4f}")
            break

    print(f"[TRAIN] Done. Saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["image_only", "metadata_only", "fusion"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_model(args.model, cfg)
