"""Fairness metrics: equalized odds gap, demographic parity gap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def compute_fairness_metrics(
    subgroup_results: pd.DataFrame,
    malignant_classes: list,
) -> pd.DataFrame:
    """Compute demographic parity gap and equalized odds gap per malignant class."""
    rows = []
    model_names = subgroup_results["model"].unique()

    for model in model_names:
        model_df = subgroup_results[subgroup_results["model"] == model]
        groups = [g for g in model_df["group"].unique() if "profile" in g]

        for cls in malignant_classes:
            group_sens = {}
            group_acc = {}
            for group in groups:
                grp_df = model_df[(model_df["group"].str.endswith(group.split("/")[-1])) & (model_df["class"] == cls)]
                if not grp_df.empty:
                    group_name = group.split("/")[-1]
                    group_sens[group_name] = grp_df["sensitivity"].values[0]
                    group_acc[group_name] = 1.0 - grp_df["fnr"].values[0]

            if len(group_sens) >= 2:
                vals = list(group_sens.values())
                eog = max(vals) - min(vals)
                dpg = max(list(group_acc.values())) - min(list(group_acc.values()))
                rows.append({
                    "model": model,
                    "class": cls,
                    "equalized_odds_gap": eog,
                    "demographic_parity_gap": dpg,
                    "group_sensitivities": str(group_sens),
                })

    return pd.DataFrame(rows)
