"""Statistical utilities for H-Plot."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t


def compute_layer_stats(
    df: pd.DataFrame,
    value_col: str,
    layer_col: str,
    distance_col: Optional[str] = None,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Compute per-layer summary statistics.

    Parameters
    ----------
    df:
        Source data containing at least ``value_col`` and ``layer_col``.
    value_col:
        Column with the metric to summarise for each layer (e.g. a proportion).
    layer_col:
        Column identifying discrete distance layers. Values are assumed to be
        sortable in the order they should appear on the x-axis.
    distance_col:
        Optional column describing the physical distance that corresponds to
        each layer. When provided, the average distance per layer is included in
        the returned dataframe. When omitted the distance column will contain
        ``NaN``.
    ci:
        Confidence interval level used to compute the bounds around the mean.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per layer containing the mean, confidence
        interval bounds, sample size and (optionally) the average physical
        distance.
    """

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' column not found in dataframe")
    if layer_col not in df.columns:
        raise ValueError(f"'{layer_col}' column not found in dataframe")
    if distance_col is not None and distance_col not in df.columns:
        raise ValueError(f"'{distance_col}' column not found in dataframe")

    grouped = df.groupby(layer_col, sort=True)
    summary = []

    for layer, group in grouped:
        values = group[value_col].dropna().to_numpy()
        n = len(values)
        if n == 0:
            continue

        mean = float(np.mean(values))
        if distance_col is not None:
            layer_distance = float(group[distance_col].dropna().mean())
        else:
            layer_distance = np.nan

        if n > 1:
            std = float(np.std(values, ddof=1))
            sem = std / np.sqrt(n)
            if n > 30:
                quantile = norm.ppf(1 - (1 - ci) / 2)
            else:
                quantile = t.ppf(1 - (1 - ci) / 2, df=n - 1)
            ci_lower = mean - quantile * sem
            ci_upper = mean + quantile * sem
        else:
            ci_lower = mean
            ci_upper = mean

        summary.append(
            {
                "layer": layer,
                "distance": layer_distance,
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
            }
        )

    return pd.DataFrame(summary).sort_values("layer").reset_index(drop=True)
