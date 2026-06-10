"""Standard metrics: AUC, balanced accuracy, per-class sensitivity, FNR, etc."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from typing import List, Optional


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def balanced_accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = None,
    label: str = "",
) -> pd.DataFrame:
    """Return per-class sensitivity, specificity, FNR, AUC + aggregate metrics."""
    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    rows = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        ppv = tp / (tp + fp + 1e-9)
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv + 1e-9)

        try:
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        except Exception:
            auc = float("nan")

        rows.append({
            "group": label,
            "class": cls,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "fnr": fnr,
            "ppv": ppv,
            "f1": f1,
            "auc": auc,
            "support": int((y_true == i).sum()),
        })

    # Aggregate
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        macro_auc = roc_auc_score(
            np.eye(n_classes)[y_true], y_prob, multi_class="ovr", average="macro"
        )
    except Exception:
        macro_auc = float("nan")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    rows.append({
        "group": label,
        "class": "AGGREGATE",
        "sensitivity": bal_acc,
        "specificity": float("nan"),
        "fnr": float("nan"),
        "ppv": float("nan"),
        "f1": macro_f1,
        "auc": macro_auc,
        "support": len(y_true),
    })

    return pd.DataFrame(rows)


def demographic_parity_gap(group_accuracies: dict) -> float:
    vals = list(group_accuracies.values())
    return max(vals) - min(vals)


def equalized_odds_gap(group_tpr: dict) -> float:
    vals = list(group_tpr.values())
    return max(vals) - min(vals)
