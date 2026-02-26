import pandas as pd
from .stats import compute_layer_stats
from .plotting import plot_hplot

class HPlot:
    def __init__(self):
        self.df_ = None
        self.grouped_stats_ = {}

    def fit(self, df, value_col, layer_col, group_col=None, distance_col=None, distance_unit=None, ci=0.95, color_map=None, palette=None, legend_order=None, legend_title=None, legend_kwargs=None):
        # Drop rows with NA in required columns
        cols = [value_col, layer_col] + ([group_col] if group_col else [])
        df = df.dropna(subset=cols)
        self.df_ = df.copy()
        self.value_col = value_col
        self.layer_col = layer_col
        self.group_col = group_col
        self.distance_col = distance_col
        self.distance_unit = distance_unit
        self.color_map = color_map
        self.palette = palette
        self.legend_order = legend_order
        self.legend_title = legend_title
        self.legend_kwargs = legend_kwargs

        if self.group_col:
            groups = df[group_col].unique()
            for group in groups:
                df_sub = df[df[group_col] == group]
                stats = compute_layer_stats(df_sub, value_col, layer_col, distance_col, ci=ci)
                self.grouped_stats_[group] = stats
        else:
            stats = compute_layer_stats(df, value_col, layer_col, distance_col, ci=ci)
            self.grouped_stats_["overall"] = stats

    def plot(self, ci_show=True, ax=None, display_base_type="tumor", display_target_type="immune cells"):
        if not self.grouped_stats_:
            raise RuntimeError("Call fit() before plot().")
        return plot_hplot(self.grouped_stats_, distance_unit=self.distance_unit, ci_show=ci_show, ax=ax, display_base_type=display_base_type, display_target_type=display_target_type, color_map=self.color_map, palette=self.palette, legend_order=self.legend_order, legend_title=self.legend_title, legend_kwargs=self.legend_kwargs)

    def savefig(self, filename, **kwargs):
        ax = self.plot()
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)
