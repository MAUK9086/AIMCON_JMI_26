"""PyTorch Dataset for ISIC 2019 — images + metadata."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.preprocessing import encode_metadata


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class ISICDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[A.Compose] = None,
        metadata_dropout_prob: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.metadata_dropout_prob = metadata_dropout_prob
        self.metadata = encode_metadata(df)
        self.labels = df["label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['image']}.jpg"

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        meta = self.metadata[idx].copy()

        # Whole-vector metadata dropout (for exp05)
        if self.metadata_dropout_prob > 0.0 and torch.rand(1).item() < self.metadata_dropout_prob:
            meta = np.zeros_like(meta)

        return {
            "image": image,
            "metadata": torch.from_numpy(meta),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": idx,
        }
