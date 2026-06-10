"""MLP metadata-only classifier."""

import torch
import torch.nn as nn
from typing import List


class MetadataOnlyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: List[int] = None,
        feature_dim: int = 64,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU()]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, image: torch.Tensor = None, metadata: torch.Tensor = None) -> torch.Tensor:
        features = self.feature_extractor(metadata)
        return self.classifier(features)

    def get_features(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(metadata)
