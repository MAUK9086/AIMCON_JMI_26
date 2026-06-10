"""Experiment 6 — Group-Aware Loss Reweighting: retrain fusion with demographic group upweighting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.preprocessing import (
    load_and_merge, impute_metadata, stratified_split, CLASS_NAMES,
    compute_class_weights, get_subgroup_mask
)
from data.dataset import ISICDataset, get_train_transforms, get_val_transforms
from evaluation.metrics import compute_full_metrics
from evaluation.subgroup_eval import evaluate_subgroups
from training.callbacks import EarlyStopping
from training.losses import build_weighted_ce_loss
from training.train import build_model, set_seed


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


def compute_group_weights(train_df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Compute per-sample weights inversely proportional to group accuracy (proxy: site+age_bucket+sex)."""
    df = train_df.copy()
    df["age_bucket"] = pd.cut(df["age_approx"], bins=[0, 30, 45, 60, 90], labels=["<30", "30-45", "45-60", "60+"])
    df["site_bucket"] = df["anatom_site_general"].apply(
        lambda s: "indian_site" if s in cfg["subgroups"]["indian_profile"]["sites"] else "western_site"
    )
    df["group_key"] = df["age_bucket"].astype(str) + "_" + df["sex"] + "_" + df["site_bucket"]

    # Simple heuristic: upweight indian_site + younger patients by 2x
    weights = np.ones(len(df), dtype=np.float32)
    indian_mask = get_subgroup_mask(df, "indian_profile", cfg)
    weights[indian_mask] = 2.0

    # Also upweight malignant classes
    malignant_idx = [CLASS_NAMES.index(c) for c in cfg["subgroups"]["malignant_classes"]]
    for idx in malignant_idx:
        weights[df["label"].values == idx] *= 1.5

    # Normalise
    weights = weights / weights.sum() * len(weights)
    return weights


def main(cfg: dict) -> None:
    print("[EXP06] Starting...")
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    for d in [tables_dir, figures_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    train_df, val_df, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    img_size = cfg["data"]["image_size"]
    bs = cfg["training"]["batch_size_fusion"]

    train_ds = ISICDataset(train_df, cfg["paths"]["images_dir"], get_train_transforms(img_size))
    val_ds = ISICDataset(val_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    test_ds = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))

    sample_weights = compute_group_weights(train_df, cfg)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Visualise group weights
    indian_mask = get_subgroup_mask(train_df, "indian_profile", cfg)
    weight_df = pd.DataFrame({"weight": sample_weights, "group": ["indian_profile" if m else "other" for m in indian_mask]})
    fig, ax = plt.subplots(figsize=(8, 4))
    weight_df.groupby("group")["weight"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Mean Sample Weight per Group")
    ax.set_ylabel("Mean Weight")
    plt.tight_layout()
    fig.savefig(figures_dir / "exp06_group_weights.png", dpi=150)
    plt.close(fig)

    # Train fusion_reweighted
    model = build_model("fusion_reweighted", cfg).to(device)
    class_weights = compute_class_weights(train_df["label"].values, cfg["models"]["num_classes"])
    criterion = build_weighted_ce_loss(class_weights, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.amp.GradScaler()

    save_path = str(models_dir / "fusion_reweighted_best.pth")
    stopper = EarlyStopping(patience=cfg["training"]["early_stopping_patience"], mode="max", save_path=save_path)

    from evaluation.metrics import balanced_accuracy_score_np

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, leave=False, desc=f"epoch {epoch+1}"):
            images = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(image=batch["image"].to(device), metadata=batch["metadata"].to(device))
                val_preds.append(logits.argmax(1).cpu().numpy())
                val_labels.append(batch["label"].numpy())
        val_bal = balanced_accuracy_score_np(np.concatenate(val_labels), np.concatenate(val_preds))
        improved = stopper.step(val_bal, model)
        print(f"  Epoch {epoch+1:02d} | val_bal={val_bal:.4f}{'  *' if improved else ''}")
        scheduler.step()
        if stopper.should_stop:
            break

    # Evaluate
    all_results = []
    for model_name in ["fusion", "fusion_dropout", "fusion_reweighted"]:
        m = build_model(model_name, cfg).to(device)
        ckpt = models_dir / f"{model_name}_best.pth"
        if not ckpt.exists():
            print(f"  [SKIP] {ckpt} not found")
            continue
        m.load_state_dict(torch.load(ckpt, map_location=device))
        y_true, y_pred, y_prob = infer(m, test_loader, device)
        sub_df = evaluate_subgroups(test_df, y_true, y_pred, y_prob, cfg, model_name)
        all_results.append(sub_df)

    result_df = pd.concat(all_results, ignore_index=True)
    result_path = tables_dir / "exp06_reweighting_comparison.csv"
    result_df.to_csv(result_path, index=False)

    print(f"[EXP06] Done. Saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
