"""Early stopping and LR scheduler callbacks."""

from __future__ import annotations
import torch
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience: int = 7, mode: str = "max", save_path: str = None):
        self.patience = patience
        self.mode = mode
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: torch.nn.Module) -> bool:
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.counter = 0
            if self.save_path is not None:
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
