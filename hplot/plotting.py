"""Plotting helpers for H-Plot."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd


GroupedStats = Dict[str, pd.DataFrame]


def plot_hplot(
    grouped_stats: GroupedStats,
    *,
    distance_unit: Optional[str] = None,
    ci_show: bool = True,
    ax: Optional[Axes] = None,
    display_base_type: str = "tumor",
    display_target_type: str = "immune cells",
) -> Axes:
    """Render the H-Plot for the provided per-group statistics."""

    if not grouped_stats:
        raise ValueError("'grouped_stats' must contain at least one group")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for label, stats_df in grouped_stats.items():
        if stats_df.empty:
            continue

        x = stats_df["layer"].round().astype(np.int32)
        ax.plot(x, stats_df["mean"], label=str(label), drawstyle="steps-post")

        if ci_show and {"ci_lower", "ci_upper"}.issubset(stats_df.columns):
            ax.fill_between(
                x,
                stats_df["ci_lower"],
                stats_df["ci_upper"],
                alpha=0.3,
                step="post",
            )

    def distance_formatter(value, _pos):
        """Formatter that appends the mean physical distance for a layer."""

        layer_index = int(round(value))
        distances = []
        for stats_df in grouped_stats.values():
            if "distance" not in stats_df.columns:
                continue
            mask = stats_df["layer"].round().astype(np.int32) == layer_index
            if mask.any():
                distances.append(stats_df.loc[mask, "distance"].dropna().mean())

        if distances:
            distance_value = float(np.mean(distances))
            if np.isnan(distance_value):
                return f"{layer_index:g}"
            if distance_unit:
                return f"{layer_index:g}\n{distance_value:.1f} {distance_unit}"
            return f"{layer_index:g}\n{distance_value:.1f}"
        return f"{layer_index:g}"

    ax.set_xlabel(
        "Layerwise cellular distance from"
        f" {display_base_type} border"
        + (
            f"\nPhysical distance ({distance_unit}) from {display_base_type} border"
            if distance_unit
            else ""
        )
    )
    ax.set_ylabel(f"Proportion of {display_target_type}")
    ax.set_title("Tumor Spatial Heterogeneity Profile (H-Plot)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(distance_formatter))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Group")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    if ax.figure:
        ax.figure.tight_layout()

    return ax
