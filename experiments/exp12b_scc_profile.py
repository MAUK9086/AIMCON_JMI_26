"""Experiment 12b — Metadata profiles per class + per-case SCC breakdown under image occlusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES
from data.dataset import ISICDataset, get_val_transforms
from training.train import build_model, get_batch_size, set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]


def infer_occluded_with_meta(model, loader, device, test_df):
    """Runs image-occluded inference, returns per-sample df with metadata attached."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    mean_image = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="occlude=image"):
            images   = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels   = batch["label"]
            images   = mean_image.expand_as(images)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs)

    y_true  = np.concatenate(all_labels)
    y_pred  = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    result = test_df[["age_approx", "anatom_site_general", "sex", "label"]].copy().reset_index(drop=True)
    result["true_label"]  = [CLASS_NAMES[i] for i in y_true]
    result["pred_label"]  = [CLASS_NAMES[i] for i in y_pred]
    for i, cn in enumerate(CLASS_NAMES):
        result[f"prob_{cn}"] = y_probs[:, i]
    return result


def main(cfg: dict) -> None:
    print("[EXP12B] Starting...")
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

    # ── Part 1: per-class metadata profiles ──
    profile_rows = []
    for cls in CLASS_NAMES:
        cls_df = test_df[test_df["label"] == CLASS_NAMES.index(cls)]
        n = len(cls_df)
        if n == 0:
            continue
        modal_site = cls_df["anatom_site_general"].mode().iloc[0] if not cls_df["anatom_site_general"].empty else "unknown"
        modal_sex  = cls_df["sex"].mode().iloc[0] if not cls_df["sex"].empty else "unknown"
        profile_rows.append({
            "class": cls,
            "n": n,
            "mean_age":      round(cls_df["age_approx"].mean(), 1),
            "median_age":    round(cls_df["age_approx"].median(), 1),
            "modal_site":    modal_site,
            "modal_site_pct": round((cls_df["anatom_site_general"] == modal_site).mean(), 3),
            "modal_sex":     modal_sex,
            "modal_sex_pct": round((cls_df["sex"] == modal_sex).mean(), 3),
        })
    profile_df = pd.DataFrame(profile_rows)
    profile_path = tables_dir / "exp12b_class_metadata_profiles.csv"
    profile_df.to_csv(profile_path, index=False)
    print(f"  Class metadata profiles saved: {profile_path}")
    print(profile_df[["class", "n", "mean_age", "modal_site", "modal_site_pct"]].to_string(index=False))

    # ── Part 2: run image-occluded inference ──
    img_size = cfg["data"]["image_size"]
    test_ds  = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    model_name = "fusion"
    bs = get_batch_size(model_name, cfg)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, cfg).to(device)
    ckpt  = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    print("  Running image-occluded inference (fusion)...")
    pred_df = infer_occluded_with_meta(model, loader, device, test_df)
    per_sample_path = tables_dir / "exp12b_per_sample_predictions.csv"
    pred_df.to_csv(per_sample_path, index=False)

    # ── Part 3: SCC case breakdown ──
    scc_df = pred_df[pred_df["true_label"] == "SCC"].copy()
    scc_df["correct_under_image_occlusion"] = scc_df["pred_label"] == "SCC"

    scc_path = tables_dir / "exp12b_scc_case_breakdown.csv"
    scc_df.to_csv(scc_path, index=False)

    correct   = scc_df[scc_df["correct_under_image_occlusion"]]
    incorrect = scc_df[~scc_df["correct_under_image_occlusion"]]
    overall_median_age = test_df["age_approx"].median()

    print(f"\n  SCC cases: {len(scc_df)} total, {len(correct)} correct, {len(incorrect)} incorrect under image occlusion")
    print(f"  Overall test-set median age: {overall_median_age:.0f}")
    for label, sub in [("Correct (metadata identified)", correct), ("Incorrect (metadata failed)", incorrect)]:
        if len(sub) == 0:
            continue
        above_median = (sub["age_approx"] > overall_median_age).mean()
        modal_s = sub["anatom_site_general"].mode().iloc[0] if not sub.empty else "N/A"
        print(f"  [{label}] n={len(sub)}, mean_age={sub['age_approx'].mean():.1f}, "
              f"above_median_age={above_median:.1%}, modal_site={modal_s}")

    print(f"\n[EXP12B] Done. Saved to: {per_sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
