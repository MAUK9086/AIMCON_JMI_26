"""Sanity checks for the pipeline — run before full overnight training."""

import sys
import numpy as np
import pandas as pd
import torch
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── Config loading ──────────────────────────────────────────────────────────

def load_cfg():
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_config_loads():
    cfg = load_cfg()
    assert cfg["project"]["seed"] == 42
    assert cfg["metadata_features"]["total_dim"] == 14
    assert cfg["models"]["num_classes"] == 8


# ── Preprocessing ───────────────────────────────────────────────────────────

def test_impute_metadata():
    from data.preprocessing import impute_metadata, CLASS_NAMES
    df = pd.DataFrame({
        "image": ["A", "B", "C"],
        "age_approx": [30.0, float("nan"), 55.0],
        "sex": ["male", float("nan"), "female"],
        "anatom_site_general": ["head/neck", float("nan"), "posterior torso"],
        "label": [0, 1, 2],
    })
    result = impute_metadata(df)
    assert result["age_approx"].isna().sum() == 0
    assert (result["sex"] == "unknown").sum() == 1
    assert (result["anatom_site_general"] == "unknown").sum() == 1
    assert result["age_norm"].between(0, 1).all()


def test_encode_metadata_shape():
    from data.preprocessing import impute_metadata, encode_metadata
    df = pd.DataFrame({
        "image": ["A", "B"],
        "age_approx": [30.0, 50.0],
        "sex": ["male", "female"],
        "anatom_site_general": ["head/neck", "lower extremity"],
        "label": [0, 1],
    })
    df = impute_metadata(df)
    feat = encode_metadata(df)
    assert feat.shape == (2, 14), f"Expected (2,14), got {feat.shape}"
    assert feat.dtype == np.float32


def test_stratified_split():
    from data.preprocessing import impute_metadata, stratified_split
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "image": [f"img_{i}" for i in range(n)],
        "age_approx": np.random.uniform(20, 80, n),
        "sex": np.random.choice(["male", "female"], n),
        "anatom_site_general": np.random.choice(["head/neck", "posterior torso", "lower extremity"], n),
        "label": np.random.randint(0, 8, n),
    })
    df = impute_metadata(df)
    train, val, test = stratified_split(df, 0.70, 0.15, 42)
    total = len(train) + len(val) + len(test)
    assert abs(total - n) <= 1
    assert len(train) > len(val)
    assert len(test) > 0


# ── Models ──────────────────────────────────────────────────────────────────

def test_image_only_forward():
    from models.image_only import ImageOnlyModel
    model = ImageOnlyModel(num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(image=x)
    assert out.shape == (2, 8)


def test_metadata_only_forward():
    from models.metadata_only import MetadataOnlyModel
    model = MetadataOnlyModel(input_dim=14, num_classes=8)
    x = torch.randn(4, 14)
    out = model(metadata=x)
    assert out.shape == (4, 8)


def test_fusion_forward():
    from models.fusion import FusionModel
    model = FusionModel(num_classes=8, metadata_input_dim=14, image_pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 14)
    out = model(image=images, metadata=metadata)
    assert out.shape == (2, 8)


# ── Losses / callbacks ───────────────────────────────────────────────────────

def test_early_stopping():
    from training.callbacks import EarlyStopping
    stopper = EarlyStopping(patience=3, mode="max")
    dummy_model = MagicMock()

    stopper.step(0.5, dummy_model)
    stopper.step(0.6, dummy_model)
    stopper.step(0.55, dummy_model)
    stopper.step(0.54, dummy_model)
    stopper.step(0.53, dummy_model)
    assert stopper.should_stop


def test_weighted_ce_loss():
    from training.losses import build_weighted_ce_loss
    weights = np.array([1.0, 2.0, 1.5, 0.8, 1.2, 3.0, 2.5, 1.0], dtype=np.float32)
    device = torch.device("cpu")
    loss_fn = build_weighted_ce_loss(weights, device)
    logits = torch.randn(4, 8)
    labels = torch.randint(0, 8, (4,))
    loss = loss_fn(logits, labels)
    assert loss.item() > 0


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_compute_full_metrics():
    from evaluation.metrics import compute_full_metrics
    np.random.seed(0)
    n = 100
    y_true = np.random.randint(0, 8, n)
    y_pred = np.random.randint(0, 8, n)
    y_prob = np.random.dirichlet(np.ones(8), n)
    df = compute_full_metrics(y_true, y_pred, y_prob, label="test")
    assert "AGGREGATE" in df["class"].values
    assert "sensitivity" in df.columns
    assert "fnr" in df.columns


def test_subgroup_mask():
    from data.preprocessing import impute_metadata, get_subgroup_mask
    import yaml
    cfg = load_cfg()
    df = pd.DataFrame({
        "image": ["A", "B", "C", "D"],
        "age_approx": [30.0, 60.0, 25.0, 55.0],
        "sex": ["female", "male", "female", "male"],
        "anatom_site_general": ["lower extremity", "posterior torso", "palms/soles", "anterior torso"],
        "label": [0, 1, 2, 3],
    })
    df = impute_metadata(df)
    indian = get_subgroup_mask(df, "indian_profile", cfg)
    western = get_subgroup_mask(df, "western_profile", cfg)
    assert indian[0] == True   # lower extremity, age 30 < 55
    assert western[1] == True  # posterior torso, age 60 >= 50, male


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
