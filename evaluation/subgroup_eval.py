"""Sliced evaluation over demographic subgroups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from evaluation.metrics import compute_full_metrics
from data.preprocessing import get_subgroup_mask, CLASS_NAMES


def evaluate_subgroups(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cfg: dict,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate metrics for all subgroups + overall."""
    results = []

    # Overall
    overall = compute_full_metrics(y_true, y_pred, y_prob, label=f"{model_name}/overall")
    overall["model"] = model_name
    results.append(overall)

    # Indian profile
    indian_mask = get_subgroup_mask(df_test, "indian_profile", cfg)
    if indian_mask.sum() >= 10:
        r = compute_full_metrics(
            y_true[indian_mask], y_pred[indian_mask], y_prob[indian_mask],
            label=f"{model_name}/indian_profile"
        )
        r["model"] = model_name
        r["n_samples"] = int(indian_mask.sum())
        results.append(r)

    # Western profile
    western_mask = get_subgroup_mask(df_test, "western_profile", cfg)
    if western_mask.sum() >= 10:
        r = compute_full_metrics(
            y_true[western_mask], y_pred[western_mask], y_prob[western_mask],
            label=f"{model_name}/western_profile"
        )
        r["model"] = model_name
        r["n_samples"] = int(western_mask.sum())
        results.append(r)

    return pd.concat(results, ignore_index=True)
