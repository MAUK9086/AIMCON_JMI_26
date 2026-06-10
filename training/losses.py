"""Weighted cross-entropy loss for class imbalance."""

import torch
import torch.nn as nn
import numpy as np


def build_weighted_ce_loss(class_weights: np.ndarray, device: torch.device) -> nn.CrossEntropyLoss:
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def build_sample_weighted_ce_loss() -> nn.CrossEntropyLoss:
    """CE loss that accepts per-sample weights via reduction='none'."""
    return nn.CrossEntropyLoss(reduction="none")
