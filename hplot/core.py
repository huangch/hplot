import pandas as pd
from .stats import compute_layer_stats
from .plotting import plot_hplot

class HPlot:
    def __init__(self):
        self.df_ = None
        self.grouped_stats_ = {}

    def fit(self, df, value_col, layer_col, region_col, group_col=None, distance_col=None, distance_unit=None, ci=0.95):
        # Drop rows with NA in required columns
        cols = [value_col, layer_col, region_col] + ([group_col] if group_col else [])
        df = df.dropna(subset=cols)
        self.df_ = df.copy()
        self.value_col = value_col
        self.layer_col = layer_col
        self.region_col = region_col
        self.group_col = group_col
        self.distance_col = distance_col
        self.distance_unit = distance_unit

        if self.group_col:
            groups = df[group_col].unique()
            for group in groups:
                df_sub = df[df[group_col] == group]
                stats = compute_layer_stats(df_sub, value_col, layer_col, region_col, distance_col, ci=ci)
                self.grouped_stats_[group] = stats
        else:
            stats = compute_layer_stats(df, value_col, layer_col, region_col)
            self.grouped_stats_["overall"] = stats

    def plot(self, ci_show=True, ax=None):
        if not self.grouped_stats_:
            raise RuntimeError("Call fit() before plot().")
        return plot_hplot(self.grouped_stats_, distance_col=self.distance_col, distance_unit = self.distance_unit, ci_show=ci_show, ax=ax)

    def savefig(self, filename, **kwargs):
        ax = self.plot()
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)