import pandas as pd
from .stats import compute_layer_stats, compute_layer_pvalues, PVALUE_TEST_LABELS
from .plotting import plot_hplot

class HPlot:
    def __init__(self):
        self.df_ = None
        self.target_grouped_stats_ = {}
        self.layer_pvalues_ = None

    def fit(self, df, targets, layer, group=None, distance=None, unit=None, ci=0.95, color_map=None, palette=None, legend_order=None, legend_title=None, legend_kwargs=None,
            pvalue=False, pvalue_test="mannwhitney", pvalue_groups=None, pvalue_correction=None, pvalue_min_n=3):
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
        self.pvalue_test = pvalue_test

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

        # Optional per-layer between-group p-value track (single target only).
        self.layer_pvalues_ = None
        if pvalue:
            if not group:
                raise ValueError("pvalue=True requires a 'group' column to compare two arms.")
            if multi:
                raise ValueError("pvalue=True supports a single target only; pass one target column.")
            self.layer_pvalues_ = compute_layer_pvalues(
                df,
                prop=target_cols[0],
                layer_col=layer,
                group_col=group,
                groups=pvalue_groups,
                test=pvalue_test,
                distance_col=distance,
                min_n=pvalue_min_n,
                correction=pvalue_correction,
            )

    def plot(self, ci_show=True, ax=None, display_base_type="tumor", display_target_type="immune cells",
             pvalue_show=False, pvalue_label=None, pvalue_color="black", pvalue_threshold=0.05,
             pvalue_threshold_show=True, pvalue_use_adjusted=False):
        if not self.target_grouped_stats_:
            raise RuntimeError("Call fit() before plot().")
        if pvalue_show and self.layer_pvalues_ is None:
            raise RuntimeError("plot(pvalue_show=True) requires fit(..., pvalue=True).")
        if pvalue_label is None:
            test_name = PVALUE_TEST_LABELS.get(getattr(self, "pvalue_test", ""), "test")
            pvalue_label = f"p-value ({test_name})"
        return plot_hplot(
            self.target_grouped_stats_, unit=self.unit, ci_show=ci_show, ax=ax,
            display_base_type=display_base_type, display_target_type=display_target_type,
            color_map=self.color_map, palette=self.palette, legend_order=self.legend_order,
            legend_title=self.legend_title, legend_kwargs=self.legend_kwargs,
            pvalue_stats=self.layer_pvalues_, pvalue_show=pvalue_show, pvalue_label=pvalue_label,
            pvalue_color=pvalue_color, pvalue_threshold=pvalue_threshold,
            pvalue_threshold_show=pvalue_threshold_show, pvalue_use_adjusted=pvalue_use_adjusted,
        )

    def savefig(self, filename, **kwargs):
        ax = self.plot()
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)
