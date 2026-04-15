import pandas as pd
from .stats import compute_layer_stats
from .plotting import plot_hplot

class HPlot:
    def __init__(self):
        self.df_ = None
        self.target_grouped_stats_ = {}

    def fit(self, df, targets, layer, group=None, distance=None, unit=None, ci=0.95, color_map=None, palette=None, legend_order=None, legend_title=None, legend_kwargs=None):
        # Normalise targets to a list
        target_cols = [targets] if isinstance(targets, str) else list(targets)
        multi = len(target_cols) > 1
        # Drop rows with NA in required columns
        cols = target_cols + [layer] + ([group] if group else [])
        df = df.dropna(subset=cols)
        self.df_ = df.copy()
        self.targets = targets
        self.layer = layer
        self.group = group
        self.distance = distance
        self.unit = unit
        self.color_map = color_map
        self.palette = palette
        self.legend_order = legend_order
        self.legend_title = legend_title
        self.legend_kwargs = legend_kwargs

        if self.group:
            groups = df[group].unique()
            for grp in groups:
                df_sub = df[df[group] == grp]
                for prop in target_cols:
                    key = f"{grp} \u2013 {prop}" if multi else grp
                    self.target_grouped_stats_[key] = compute_layer_stats(df_sub, prop, layer, distance, ci=ci)
        else:
            for prop in target_cols:
                key = prop if multi else "overall"
                self.target_grouped_stats_[key] = compute_layer_stats(df, prop, layer, distance, ci=ci)

    def plot(self, ci_show=True, ax=None, display_base_type="tumor", display_target_type="immune cells"):
        if not self.target_grouped_stats_:
            raise RuntimeError("Call fit() before plot().")
        return plot_hplot(self.target_grouped_stats_, unit=self.unit, ci_show=ci_show, ax=ax, display_base_type=display_base_type, display_target_type=display_target_type, color_map=self.color_map, palette=self.palette, legend_order=self.legend_order, legend_title=self.legend_title, legend_kwargs=self.legend_kwargs)

    def savefig(self, filename, **kwargs):
        ax = self.plot()
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)
