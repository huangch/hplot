"""Execution helpers for batch H-Plot generation."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .core import HPlot


def run_hplot_batch(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    layer_col: str = "layer",
    group_col: Optional[str] = None,
    case_col: Optional[str] = None,
    distance_col: Optional[str] = None,
    distance_unit: Optional[str] = None,
    ci: float = 0.95,
    output_dir: str = "hplots",
    file_prefix: str = "hplot",
    ci_show: bool = True,
    file_format: str = "svg",
    dpi: int = 300,
    display_base_type: str = "tumor",
    display_target_type: str = "immune cells",
) -> None:
    """Generate H-Plots for each ``case_col`` value.

    If ``case_col`` is ``None`` the entire dataframe is visualised in a single
    plot. When provided, a separate file is written for each case.
    """

    if file_format not in {"svg", "pdf", "png"}:
        raise ValueError("file_format must be one of 'svg', 'pdf' or 'png'")

    os.makedirs(output_dir, exist_ok=True)

    if case_col and case_col in df.columns:
        case_groups = sorted(df[case_col].dropna().unique())
    else:
        case_groups = [None]

    for case_value in case_groups:
        if case_value is None:
            subset = df
            suffix = ""
        else:
            subset = df[df[case_col] == case_value]
            if subset.empty:
                continue
            suffix = f"_{case_value}"

        hplot = HPlot()
        hplot.fit(
            subset,
            value_col=value_col,
            layer_col=layer_col,
            group_col=group_col,
            distance_col=distance_col,
            distance_unit=distance_unit,
            ci=ci,
        )

        if not hplot.grouped_stats_:
            continue

        filename = os.path.join(output_dir, f"{file_prefix}{suffix}.{file_format}")
        hplot.savefig(
            filename,
            ci_show=ci_show,
            display_base_type=display_base_type,
            display_target_type=display_target_type,
            dpi=dpi,
            format=file_format,
        )
