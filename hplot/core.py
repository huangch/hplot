"""High-level API for fitting and plotting H-Plots."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from matplotlib.axes import Axes

from .plotting import plot_hplot
from .stats import compute_layer_stats


class HPlot:
    """Compute and render H-Plots from tabular data."""

    def __init__(self) -> None:
        self.df_: Optional[pd.DataFrame] = None
        self.grouped_stats_: Dict[str, pd.DataFrame] = {}
        self.value_col: Optional[str] = None
        self.layer_col: Optional[str] = None
        self.group_col: Optional[str] = None
        self.distance_col: Optional[str] = None
        self.distance_unit: Optional[str] = None
        self.ci: float = 0.95

    def fit(
        self,
        df: pd.DataFrame,
        *,
        value_col: str,
        layer_col: str,
        group_col: Optional[str] = None,
        distance_col: Optional[str] = None,
        distance_unit: Optional[str] = None,
        ci: float = 0.95,
    ) -> None:
        """Compute layer statistics grouped by ``group_col``.

        The dataframe must contain the columns referenced by ``value_col`` and
        ``layer_col``. Rows with missing values in those columns are dropped.
        When ``group_col`` is provided, a separate curve is generated per group
        value; otherwise a single curve named ``"overall"`` is produced.
        """

        required_columns = {value_col, layer_col}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")
        if group_col and group_col not in df.columns:
            raise ValueError(f"'{group_col}' column not found in dataframe")
        if distance_col and distance_col not in df.columns:
            raise ValueError(f"'{distance_col}' column not found in dataframe")

        clean_df = df.dropna(subset=list(required_columns)).copy()

        self.df_ = clean_df
        self.value_col = value_col
        self.layer_col = layer_col
        self.group_col = group_col
        self.distance_col = distance_col
        self.distance_unit = distance_unit
        self.ci = ci
        self.grouped_stats_.clear()

        if group_col:
            for group_value, grouped_df in clean_df.groupby(group_col):
                stats = compute_layer_stats(
                    grouped_df,
                    value_col=value_col,
                    layer_col=layer_col,
                    distance_col=distance_col,
                    ci=ci,
                )
                if not stats.empty:
                    self.grouped_stats_[str(group_value)] = stats
        else:
            stats = compute_layer_stats(
                clean_df,
                value_col=value_col,
                layer_col=layer_col,
                distance_col=distance_col,
                ci=ci,
            )
            if not stats.empty:
                self.grouped_stats_["overall"] = stats

    def plot(
        self,
        *,
        ci_show: bool = True,
        ax: Optional[Axes] = None,
        display_base_type: str = "tumor",
        display_target_type: str = "immune cells",
    ):
        """Plot the fitted H-Plot and return the matplotlib axes object."""

        if not self.grouped_stats_:
            raise RuntimeError("Call fit() before plot().")

        return plot_hplot(
            self.grouped_stats_,
            distance_unit=self.distance_unit,
            ci_show=ci_show,
            ax=ax,
            display_base_type=display_base_type,
            display_target_type=display_target_type,
        )

    def savefig(
        self,
        filename: str,
        *,
        ci_show: bool = True,
        display_base_type: str = "tumor",
        display_target_type: str = "immune cells",
        **kwargs,
    ) -> None:
        """Render the plot and save it to ``filename``."""

        ax = self.plot(
            ci_show=ci_show,
            display_base_type=display_base_type,
            display_target_type=display_target_type,
        )
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)
