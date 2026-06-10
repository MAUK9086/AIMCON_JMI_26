"""Metadata encoding, imputation, and train/val/test split logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

SITE_CATEGORIES = [
    "anterior torso",
    "head/neck",
    "lateral torso",
    "lower extremity",
    "oral/genital",
    "palms/soles",
    "posterior torso",
    "upper extremity",
    "unknown",
]

SEX_CATEGORIES = ["male", "female", "unknown"]


def load_and_merge(metadata_csv: str, ground_truth_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(ground_truth_csv)
    df = meta.merge(gt, on="image")
    df["label"] = df[CLASS_NAMES].values.argmax(axis=1)
    return df


def impute_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_median = df["age_approx"].median()
    df["age_approx"] = df["age_approx"].fillna(age_median)
    df["sex"] = df["sex"].fillna("unknown").str.lower()
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown").str.lower()
    # Normalise age to [0, 1] based on observed range
    age_min, age_max = 0.0, 90.0
    df["age_norm"] = (df["age_approx"] - age_min) / (age_max - age_min)
    df["age_norm"] = df["age_norm"].clip(0.0, 1.0)
    return df


def encode_metadata(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 14) float32 feature matrix: [age_norm(1), sex_onehot(3), site_onehot(10)]."""
    n = len(df)
    features = np.zeros((n, 14), dtype=np.float32)

    # Age
    features[:, 0] = df["age_norm"].values

    # Sex one-hot (3 dims: male, female, unknown)
    for i, cat in enumerate(SEX_CATEGORIES):
        mask = df["sex"].values == cat
        features[mask, 1 + i] = 1.0

    # Site one-hot (10 dims: 8 named + unknown + catch-all unknown)
    # Map any unseen value to "unknown"
    site_vals = df["anatom_site_general"].values
    for i, cat in enumerate(SITE_CATEGORIES):
        mask = site_vals == cat
        features[mask, 4 + i] = 1.0

    # Anything not matched goes to the unknown slot (index 4+8=12)
    known_sites = set(SITE_CATEGORIES[:-1])
    for j, sv in enumerate(site_vals):
        if sv not in known_sites and sv != "unknown":
            features[j, 4 + 8] = 1.0  # unknown slot

    return features


def stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_frac = 1.0 - train_frac - val_frac
    train_df, tmp = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label"], random_state=seed
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        tmp, test_size=relative_test, stratify=tmp["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32)


def get_subgroup_mask(df: pd.DataFrame, profile: str, cfg: dict) -> np.ndarray:
    """Return boolean mask for indian_profile or western_profile."""
    if profile == "indian_profile":
        p = cfg["subgroups"]["indian_profile"]
        mask = (
            df["anatom_site_general"].isin(p["sites"])
            & (df["age_approx"] < p["max_age"])
        )
    elif profile == "western_profile":
        p = cfg["subgroups"]["western_profile"]
        mask = (
            df["anatom_site_general"].isin(p["sites"])
            & (df["age_approx"] >= p["min_age"])
            & (df["sex"] == p["sex"])
        )
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return mask.values
