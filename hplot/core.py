import numpy as np
import pandas as pd
from .stats import compute_layer_stats, compute_layer_pvalues, PVALUE_TEST_LABELS
from .plotting import plot_hplot


def _gam_curve_to_stats_df(grid, pred, ci, dist_by_layer, n_by_layer):
    """Normalise a per-group GAM smooth to the compute_layer_stats schema.

    Returns a DataFrame with the same columns produced by
    :func:`hplot.stats.compute_layer_stats` (``layer, distance, mean,
    ci_lower, ci_upper, n``) so that ``plot()`` can render a GAM smooth and a
    layer-binned mean identically.
    """
    grid = np.asarray(grid, dtype=float)
    ci = np.asarray(ci, dtype=float)
    if dist_by_layer is not None:
        distance = dist_by_layer.reindex(grid).to_numpy()
    else:
        distance = np.full(grid.shape, np.nan)
    if n_by_layer is not None:
        n = n_by_layer.reindex(grid).fillna(0).to_numpy()
    else:
        n = np.full(grid.shape, np.nan)
    return pd.DataFrame({
        "layer": grid,
        "distance": distance,
        "mean": np.asarray(pred, dtype=float),
        "ci_lower": ci[:, 0],
        "ci_upper": ci[:, 1],
        "n": n,
    })


class HPlot:
    def __init__(self):
        self.df_ = None
        self.target_grouped_stats_ = {}
        self.layer_pvalues_ = None
        self.gam_curves_ = {}

    def fit(self, df, targets, layer, group=None, distance=None, unit=None, ci=0.95, color_map=None, palette=None, legend_order=None, legend_title=None, legend_kwargs=None,
            pvalue=False, pvalue_test="mannwhitney", pvalue_groups=None, pvalue_correction=None, pvalue_min_n=3,
            smoother="mean", gam_grid=None, gam_n_splines=10, gam_lam_grid=None, gam_ci_width=0.95, gam_group_order=None):
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

        if smoother not in ("mean", "gam"):
            raise ValueError(f"Unknown smoother={smoother!r}; use 'mean' or 'gam'.")

        # Reset any state from a previous fit() so repeated calls are clean.
        self.target_grouped_stats_ = {}
        self.gam_curves_ = {}

        if smoother == "mean":
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
        else:  # smoother == "gam"
            from .stats import gam_group_curves
            grid = gam_grid
            if grid is None:
                grid = np.sort(df[layer].dropna().unique()).astype(float)
            grid = np.asarray(grid, dtype=float)
            dist_by_layer = df.groupby(layer)[distance].mean() if distance else None
            for prop in target_cols:
                if self.group:
                    gorder = list(gam_group_order) if gam_group_order is not None else list(pd.unique(df[group].dropna()))
                    curves = gam_group_curves(
                        df, prop, layer, group, grid, groups=tuple(gorder),
                        n_splines=gam_n_splines, lam_grid=gam_lam_grid, ci_width=gam_ci_width,
                    )
                    self.gam_curves_[prop] = curves
                    for grp, (pred, ci_arr) in curves.items():
                        key = f"{grp} \u2013 {prop}" if multi else grp
                        n_by_layer = df[df[group] == grp].groupby(layer)[prop].count()
                        self.target_grouped_stats_[key] = _gam_curve_to_stats_df(
                            grid, pred, ci_arr, dist_by_layer, n_by_layer)
                else:
                    tmp = df.assign(_hplot_grp="overall")
                    curves = gam_group_curves(
                        tmp, prop, layer, "_hplot_grp", grid, groups=("overall",),
                        n_splines=gam_n_splines, lam_grid=gam_lam_grid, ci_width=gam_ci_width,
                    )
                    self.gam_curves_[prop] = curves
                    pred, ci_arr = curves["overall"]
                    key = prop if multi else "overall"
                    n_by_layer = df.groupby(layer)[prop].count()
                    self.target_grouped_stats_[key] = _gam_curve_to_stats_df(
                        grid, pred, ci_arr, dist_by_layer, n_by_layer)


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
             value_kind="proportion", ylabel=None,
             pvalue_show=False, pvalue_label=None, pvalue_color="black", pvalue_threshold=0.05,
             pvalue_threshold_show=True, pvalue_use_adjusted=False, pvalue_ylim=None,
             band=None, band_threshold=None, band_min_width=2, band_color="0.6",
             band_alpha=0.12, band_label=None):
        if not self.target_grouped_stats_:
            raise RuntimeError("Call fit() before plot().")
        if pvalue_show and self.layer_pvalues_ is None:
            raise RuntimeError("plot(pvalue_show=True) requires fit(..., pvalue=True).")
        if isinstance(band, str) and band == "auto" and self.layer_pvalues_ is None:
            raise RuntimeError("plot(band='auto') requires fit(..., pvalue=True).")
        if pvalue_label is None:
            test_name = PVALUE_TEST_LABELS.get(getattr(self, "pvalue_test", ""), "test")
            pvalue_label = f"p-value ({test_name})"
        return plot_hplot(
            self.target_grouped_stats_, unit=self.unit, ci_show=ci_show, ax=ax,
            display_base_type=display_base_type, display_target_type=display_target_type,
            value_kind=value_kind, ylabel=ylabel,
            color_map=self.color_map, palette=self.palette, legend_order=self.legend_order,
            legend_title=self.legend_title, legend_kwargs=self.legend_kwargs,
            pvalue_stats=self.layer_pvalues_, pvalue_show=pvalue_show, pvalue_label=pvalue_label,
            pvalue_color=pvalue_color, pvalue_threshold=pvalue_threshold,
            pvalue_threshold_show=pvalue_threshold_show, pvalue_use_adjusted=pvalue_use_adjusted,
            pvalue_ylim=pvalue_ylim,
            band=band, band_threshold=band_threshold, band_min_width=band_min_width,
            band_color=band_color, band_alpha=band_alpha, band_label=band_label,
        )

    def savefig(self, filename, **kwargs):
        ax = self.plot()
        fig = ax.get_figure()
        fig.savefig(filename, **kwargs)

    def gam_delta(self, target=None, groups=None):
        """Pointwise Δ(layer) = high − low from the stored GAM smooths.

        Requires a prior ``fit(..., smoother="gam", group=...)`` with two
        groups. Thin wrapper around :func:`hplot.stats.gam_delta_curve`.

        Returns ``(diff_pred, ci_lower, ci_upper, sig_pos, sig_neg)``.
        """
        from .stats import gam_delta_curve
        if not self.gam_curves_:
            raise RuntimeError("gam_delta() requires fit(..., smoother='gam').")
        if target is None:
            if len(self.gam_curves_) != 1:
                raise ValueError(
                    "Multiple targets fitted; pass target= to pick one of "
                    f"{list(self.gam_curves_)}.")
            target = next(iter(self.gam_curves_))
        return gam_delta_curve(self.gam_curves_[target], groups=groups)
