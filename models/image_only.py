"""EfficientNet-B3 image-only classifier."""

import timm
import torch
import torch.nn as nn


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
            global_pool="avg",
        )
        feature_dim = self.encoder.num_features  # 1536 for B3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features)

    def get_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)
