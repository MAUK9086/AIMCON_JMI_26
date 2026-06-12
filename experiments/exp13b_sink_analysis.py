"""Experiment 13b — AK/DF sink ratio analysis and sink region metadata profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.preprocessing import load_and_merge, impute_metadata, stratified_split, CLASS_NAMES


def main(cfg: dict) -> None:
    print("[EXP13B] Starting...")

    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load per-sample predictions saved by exp13
    pred_path = tables_dir / "exp13_per_sample_predictions.csv"
    if not pred_path.exists():
        # Fall back: reconstruct from confusion matrix + test_df
        print("  per_sample_predictions not found — reconstructing from confusion matrix")
        pred_path = None

    seed = cfg["project"]["seed"]
    df = load_and_merge(cfg["paths"]["metadata_csv"], cfg["paths"]["ground_truth_csv"])
    df = impute_metadata(df)
    _, _, test_df = stratified_split(df, cfg["data"]["train_split"], cfg["data"]["val_split"], seed)

    # We need per-sample predictions for exp13. Check if exp12b also saved them.
    # exp13 does not save per-sample predictions by default — re-run a quick metadata_only inference.
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from data.dataset import ISICDataset, get_val_transforms
    from training.train import build_model, get_batch_size, set_seed

    set_seed(seed)
    device = torch.device(cfg["project"]["device"])

    img_size = cfg["data"]["image_size"]
    test_ds  = ISICDataset(test_df, cfg["paths"]["images_dir"], get_val_transforms(img_size))
    model_name = "metadata_only"
    bs = get_batch_size(model_name, cfg)
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, cfg).to(device)
    ckpt  = Path(cfg["paths"]["models_dir"]) / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="metadata_only infer"):
            images   = batch["image"].to(device)
            metadata = batch["metadata"].to(device)
            labels   = batch["label"]
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(image=images, metadata=metadata)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    n_total = len(y_true)

    # ── Part 1: sink ratios ──
    true_counts = np.bincount(y_true, minlength=len(CLASS_NAMES))
    pred_counts = np.bincount(y_pred, minlength=len(CLASS_NAMES))
    sink_rows = []
    for i, cls in enumerate(CLASS_NAMES):
        true_prop = true_counts[i] / n_total
        pred_prop = pred_counts[i] / n_total
        sink_ratio = pred_prop / (true_prop + 1e-9)
        sink_rows.append({
            "class": cls,
            "true_count": int(true_counts[i]),
            "predicted_count": int(pred_counts[i]),
            "true_proportion": round(true_prop, 4),
            "predicted_proportion": round(pred_prop, 4),
            "sink_ratio": round(sink_ratio, 3),
        })
    sink_df = pd.DataFrame(sink_rows).sort_values("sink_ratio", ascending=False)
    sink_path = tables_dir / "exp13b_sink_ratios.csv"
    sink_df.to_csv(sink_path, index=False)
    print(f"\n  Sink ratios:")
    print(sink_df.to_string(index=False))

    # ── Part 2: sink region metadata profiles ──
    test_meta = test_df[["age_approx", "anatom_site_general", "sex", "label"]].copy().reset_index(drop=True)
    test_meta["pred_label"] = [CLASS_NAMES[i] for i in y_pred]
    test_meta["true_label"] = [CLASS_NAMES[i] for i in y_true]

    profile_rows = []
    # Profiles for: predicted-as-AK, predicted-as-DF, true-AK, true-DF, overall
    groups = {
        "predicted_as_AK":  test_meta[test_meta["pred_label"] == "AK"],
        "predicted_as_DF":  test_meta[test_meta["pred_label"] == "DF"],
        "true_AK":          test_meta[test_meta["true_label"] == "AK"],
        "true_DF":          test_meta[test_meta["true_label"] == "DF"],
        "overall_test_set": test_meta,
    }
    for grp_name, sub in groups.items():
        if len(sub) == 0:
            continue
        modal_site = sub["anatom_site_general"].mode().iloc[0] if not sub["anatom_site_general"].empty else "unknown"
        modal_sex  = sub["sex"].mode().iloc[0] if not sub["sex"].empty else "unknown"
        profile_rows.append({
            "group": grp_name,
            "n": len(sub),
            "mean_age":      round(sub["age_approx"].mean(), 1),
            "median_age":    round(sub["age_approx"].median(), 1),
            "modal_site":    modal_site,
            "modal_site_pct": round((sub["anatom_site_general"] == modal_site).mean(), 3),
            "modal_sex":     modal_sex,
            "modal_sex_pct": round((sub["sex"] == modal_sex).mean(), 3),
        })
    profile_df = pd.DataFrame(profile_rows)
    profile_path = tables_dir / "exp13b_sink_profiles.csv"
    profile_df.to_csv(profile_path, index=False)
    print(f"\n  Sink region profiles:")
    print(profile_df.to_string(index=False))

    print(f"\n[EXP13B] Done. Saved to: {sink_path}, {profile_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
