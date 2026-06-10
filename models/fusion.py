"""Late fusion: EfficientNet-B3 image features + MLP metadata features."""

import timm
import torch
import torch.nn as nn
from typing import List


class FusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        image_feature_dim: int = 1536,
        metadata_input_dim: int = 14,
        metadata_hidden_dims: List[int] = None,
        metadata_feature_dim: int = 64,
        fusion_hidden_dims: List[int] = None,
        dropout: float = 0.3,
        image_pretrained: bool = True,
    ):
        super().__init__()
        if metadata_hidden_dims is None:
            metadata_hidden_dims = [64, 128, 64]
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [256]

        # Image branch
        self.image_encoder = timm.create_model(
            "efficientnet_b3",
            pretrained=image_pretrained,
            num_classes=0,
            global_pool="avg",
        )
        image_feature_dim = self.image_encoder.num_features

        # Metadata branch
        meta_layers = []
        in_dim = metadata_input_dim
        for h in metadata_hidden_dims:
            meta_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        meta_layers += [nn.Linear(in_dim, metadata_feature_dim), nn.BatchNorm1d(metadata_feature_dim), nn.ReLU()]
        self.metadata_encoder = nn.Sequential(*meta_layers)

        # Fusion head
        fusion_in = image_feature_dim + metadata_feature_dim
        fusion_layers = []
        for h in fusion_hidden_dims:
            fusion_layers += [nn.Linear(fusion_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            fusion_in = h
        fusion_layers.append(nn.Linear(fusion_in, num_classes))
        self.fusion_head = nn.Sequential(*fusion_layers)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        meta_feat = self.metadata_encoder(metadata)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion_head(combined)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def get_metadata_features(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.metadata_encoder(metadata)
