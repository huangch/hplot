import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Column geometry of the H-Pathway colour-key row, as a fraction of the key
# axes. Shared by the layout pass (which fits the column titles to this width)
# and the drawing pass.
_KEY_CB_W, _KEY_GAP = 0.18, 0.10

# Phrasing templates for the y-axis label, keyed by the kind of quantity the
# H-plot is showing. The y-value is always a per-layer summary, but its meaning
# differs by target mode:
#   - "proportion": fraction of cells of a given cell type (0..1).
#   - "fraction":   fraction of cells in a given niche / CME (0..1).
#   - "expression": mean expression level of a gene/signature (a.u., unbounded).
#   - "interaction": mean ligand-receptor interaction score (a.u., unbounded);
#     for cell-cell interaction (CCI) targets such as "CCL19->CCR7".
_VALUE_KIND_TEMPLATES = {
   "proportion": "Proportion of {target}",
   "fraction": "Fraction of cells in {target}",
   "expression": "Mean expression of {target}",
   "interaction": "Mean interaction score of {target}",
}


def _build_ylabel(value_kind, display_target_type, ylabel=None):
   """Resolve the y-axis label.

   An explicit ``ylabel`` always wins. Otherwise the label is composed from
   ``value_kind`` (see ``_VALUE_KIND_TEMPLATES``) and ``display_target_type``.
   """
   if ylabel is not None:
       return ylabel
   try:
       template = _VALUE_KIND_TEMPLATES[value_kind]
   except KeyError:
       raise ValueError(
           f"Unknown value_kind={value_kind!r}; expected one of "
           f"{sorted(_VALUE_KIND_TEMPLATES)} or an explicit ylabel."
       )
   return template.format(target=display_target_type)


def _contiguous_significant_bands(layers, pvals, threshold, min_width):
   """Find contiguous layer ranges that pass a per-layer p-value threshold.

   A band is a maximal run of consecutive tested layers whose p-value is
   finite and below ``threshold``. Only runs spanning at least ``min_width``
   layers are returned: an isolated single-layer spike is ignored because real
   spatial biology is smooth and lone blips are the signature of noise. This
   mirrors the cluster-forming step of a cluster-mass spatial screen.

   Parameters
   ----------
   layers : array-like
       Layer indices (need not be sorted).
   pvals : array-like
       Per-layer p-values aligned with ``layers``; NaN/inf break a run.
   threshold : float
       Per-layer significance threshold (e.g. 0.05).
   min_width : int
       Minimum number of consecutive significant layers for a band to count.

   Returns
   -------
   list[tuple[float, float]]
       ``(lo, hi)`` layer ranges, one per qualifying band.
   """
   layers = np.asarray(layers, dtype=float)
   pvals = np.asarray(pvals, dtype=float)
   order = np.argsort(layers)
   layers = layers[order]
   pvals = pvals[order]
   sig = np.isfinite(pvals) & (pvals < threshold)
   bands = []
   start = None
   for i, s in enumerate(sig):
       if s and start is None:
           start = i
       elif not s and start is not None:
           if i - start >= min_width:
               bands.append((float(layers[start]), float(layers[i - 1])))
           start = None
   if start is not None and len(sig) - start >= min_width:
       bands.append((float(layers[start]), float(layers[-1])))
   return bands


def plot_hplot(
   target_grouped_stats,
   unit=None,
   ci_show=True,
   ax=None,
   display_base_type="tumor",
   display_target_type="immune cells",
   value_kind="proportion",
   ylabel=None,
   color_map=None,
   palette=None,
   legend_order=None,
   legend_title="Group",
   legend_kwargs=None,
   pvalue_stats=None,
   pvalue_show=False,
   pvalue_label="p-value",
   pvalue_color="black",
   pvalue_threshold=0.05,
   pvalue_threshold_show=True,
   pvalue_use_adjusted=False,
   pvalue_ylim=None,
   band=None,
   band_threshold=None,
   band_min_width=2,
   band_color="0.6",
   band_alpha=0.12,
   band_label=None,
   gam_curves=None,
   gam_curves_ci_show=True,
   gam_curves_linestyle="--",
   gam_curves_linewidth=1.8,
   gam_curves_ci_alpha=0.10,
   gam_curves_grid=None,
   gam_curves_label_suffix=" (GAM)",
):
   """
   Plot H-plot curves from precomputed grouped_stats.
   Parameters
   ----------
   target_grouped_stats : dict[str, pd.DataFrame]
       Mapping from group label -> stats DataFrame.
       DataFrame must contain columns: 'layer', 'mean' and (if ci_show) 'ci_lower', 'ci_upper'.
       If distance tick labels are desired, DataFrame should contain column: 'distance'.
   unit : str | None
       Unit string shown on x tick second line (optional).
   ci_show : bool
       Whether to draw confidence interval bands using fill_between.
   ax : matplotlib.axes.Axes | None
       Existing axis to draw into; if None, create a new figure/axis.
   display_base_type : str
       Used only for title text.
   display_target_type : str
       Target name interpolated into the y-axis label.
   value_kind : str
       Kind of quantity on the y-axis: 'proportion' (cell-type fraction),
       'fraction' (niche/CME fraction) or 'expression' (mean gene expression).
       Selects the y-axis label phrasing. Ignored when ``ylabel`` is given.
   ylabel : str | None
       Explicit y-axis label. Overrides the ``value_kind`` template entirely.
   color_map : dict[str, str] | None
       Explicit mapping label -> matplotlib color. If provided, overrides palette.
   palette : sequence | None
       Sequence of colors used when color_map is None. Defaults to plt.cm.tab10.colors.
   legend_order : list[str] | None
       If provided, legend entries are shown in this order (labels not present are ignored).
   legend_title : str
       Title for legend box.
   legend_kwargs : dict | None
       Extra kwargs forwarded to ax.legend(...).
   pvalue_stats : pd.DataFrame | None
       Per-layer p-value table from compute_layer_pvalues (columns 'layer',
       'p_value', optionally 'p_adj'). Required when pvalue_show is True.
   pvalue_show : bool
       Draw the per-layer p-value as a dashed line on a secondary log y-axis.
       The p-axis is only created when at least one layer has a valid
       (finite) p-value; if every layer is untestable the axis is skipped
       entirely rather than left empty.
   pvalue_label : str
       Y-axis label / legend entry for the p-value track.
   pvalue_color : str
       Colour of the p-value line and reference threshold.
   pvalue_threshold : float
       Significance level drawn as a horizontal reference line.
   pvalue_threshold_show : bool
       Whether to draw the threshold reference line.
   pvalue_use_adjusted : bool
       Plot the multiple-testing-corrected 'p_adj' column instead of 'p_value'.
   pvalue_ylim : tuple[float, float] | None
       Explicit ``(bottom, top)`` for the secondary p-value log axis. When
       ``None`` (default) the range is auto-scaled per panel so the whole
       p-curve and the threshold line stay in-frame. Pass a fixed range to
       make p-axes comparable across panels of a multi-panel figure.
   band : None | tuple | list[tuple] | "auto"
       Shaded vertical band(s) highlighting spatially significant layer
       ranges. Pass an explicit ``(lo, hi)`` layer range, a list of such
       ranges, or the string ``"auto"`` to derive contiguous significant
       band(s) from ``pvalue_stats`` (a maximal run of layers with
       p < ``band_threshold`` spanning at least ``band_min_width`` layers).
       ``"auto"`` requires ``pvalue_stats`` to be supplied. Bands are drawn
       behind the curves (zorder=0).
   band_threshold : float | None
       Per-layer p-value cutoff used when ``band="auto"``. Defaults to
       ``pvalue_threshold`` when ``None``.
   band_min_width : int
       Minimum number of consecutive significant layers for an auto band
       (single-layer spikes are ignored). Default 2.
   band_color : str
       Fill colour of the shaded band(s). Default mid-grey ``"0.6"``.
   band_alpha : float
       Opacity of the shaded band(s). Default 0.12.
   band_label : str | None
       Legend label for the band(s); only the first span is labelled so the
       legend has a single entry. ``None`` keeps the band out of the legend.
   gam_curves : dict | None
       GAM smooth-curve overlay from :func:`hplot.stats.gam_group_curves`.
       Expected format: ``{group_label: (pred_array, ci_array)}`` where
       *pred_array* is shape ``(G,)`` and *ci_array* is shape ``(G, 2)``.
       Each group is drawn on top of the raw-mean curve in the same colour
       using ``gam_curves_linestyle``.  Pass ``None`` to skip (default).
   gam_curves_ci_show : bool
       Whether to shade the GAM 95 % pointwise CI band.  Default ``True``.
   gam_curves_linestyle : str
       Matplotlib linestyle for the GAM smooth line.  Default ``"--"``.
   gam_curves_linewidth : float
       Line width for the GAM smooth line.  Default ``1.8``.
   gam_curves_ci_alpha : float
       Opacity of the GAM CI shading (lighter than the raw-mean CI so the
       two are visually distinct).  Default ``0.10``.
   gam_curves_grid : array-like | None
       X-coordinates (layer values) that correspond to the rows of the
       prediction arrays in ``gam_curves``.  Required when the GAM grid is
       not aligned with the integer layer indices in ``target_grouped_stats``.
       When ``None`` the function assumes the GAM grid matches the sorted
       integer layers found in the first group of ``target_grouped_stats``.
   gam_curves_label_suffix : str
       Text appended to the group label in the legend for GAM curve entries.
       Set to ``""`` to suppress a separate legend entry.  Default
       ``" (GAM)"``.
   """
   if legend_kwargs is None:
       legend_kwargs = {}
   # default palette if neither specified
   if color_map is None and palette is None:
       palette = plt.cm.tab10.colors

   if ax is None:
       _, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", (6, 4)))
   if True:
       # Plot each group
       for i, (label, df) in enumerate(target_grouped_stats.items()):
           x = df["layer"].round().astype(np.int32).to_numpy()
           y = df["mean"].to_numpy()
           if color_map is not None:
               if label not in color_map:
                   raise ValueError(f"Missing color for label '{label}' in color_map.")
               color = color_map[label]
           else:
               color = palette[i % len(palette)]
           ax.plot(
               x,
               y,
               label=str(label),
               color=color,
               linewidth=2,
           )
           if ci_show:
               if ("ci_lower" not in df.columns) or ("ci_upper" not in df.columns):
                   raise ValueError(
                       f"ci_show=True but '{label}' stats missing ci_lower/ci_upper columns."
                   )
               ax.fill_between(
                   x,
                   df["ci_lower"].to_numpy(),
                   df["ci_upper"].to_numpy(),
                   color=color,
                   alpha=0.25,
               )

       # Optional GAM smooth-curve overlay (Stage-2 confounder-adjusted curves)
       if gam_curves is not None:
           # Resolve the x-grid for the GAM predictions.
           if gam_curves_grid is not None:
               gam_x = np.asarray(gam_curves_grid, dtype=float)
           else:
               # Fall back to sorted integer layers from the first group.
               first_df = next(iter(target_grouped_stats.values()))
               gam_x = np.sort(first_df["layer"].round().astype(np.int32).to_numpy())
           # Determine color index per group (mirrors raw-curve loop above).
           group_color = {}
           for i, label in enumerate(target_grouped_stats):
               if color_map is not None:
                   group_color[label] = color_map.get(label)
               else:
                   group_color[label] = palette[i % len(palette)]
           for grp_label, (pred, ci) in gam_curves.items():
               color = group_color.get(grp_label)
               if color is None:
                   # GAM group not in target_grouped_stats — pick next palette slot
                   color = palette[len(group_color) % len(palette)]
               legend_label = (
                   f"{grp_label}{gam_curves_label_suffix}"
                   if gam_curves_label_suffix
                   else None
               )
               ax.plot(
                   gam_x,
                   pred,
                   color=color,
                   linestyle=gam_curves_linestyle,
                   linewidth=gam_curves_linewidth,
                   label=legend_label,
               )
               if gam_curves_ci_show and ci is not None:
                   ax.fill_between(
                       gam_x,
                       ci[:, 0],
                       ci[:, 1],
                       color=color,
                       alpha=gam_curves_ci_alpha,
                   )


       ax.set_ylabel(_build_ylabel(value_kind, display_target_type, ylabel))
       ax.set_title(f"{display_base_type.capitalize()} Spatial Heterogeneity Profile (H-plot)", fontweight="bold")
       ax.tick_params(axis="both")
       ax.grid(True, linestyle="--", alpha=0.5)
       ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

       # Optional shaded "significant band(s)": contiguous layer ranges that
       # carry the signal. Either supplied explicitly (e.g. the winning band of
       # an external cluster-mass screen) or derived here from the per-layer
       # p-value track via band="auto".
       if band is not None:
           if isinstance(band, str):
               if band != "auto":
                   raise ValueError(
                       f"band={band!r} not understood; use a (lo, hi) tuple, "
                       f"a list of tuples, or 'auto'."
                   )
               if pvalue_stats is None or len(pvalue_stats) == 0:
                   raise ValueError(
                       "band='auto' requires pvalue_stats (fit(..., pvalue=True))."
                   )
               bcol = "p_adj" if pvalue_use_adjusted else "p_value"
               if bcol not in pvalue_stats.columns:
                   raise ValueError(f"pvalue_stats missing '{bcol}' column for band='auto'.")
               thr = pvalue_threshold if band_threshold is None else band_threshold
               bstats = pvalue_stats.sort_values("layer")
               band_spans = _contiguous_significant_bands(
                   bstats["layer"].to_numpy(), bstats[bcol].to_numpy(),
                   thr, band_min_width,
               )
           else:
               # Explicit (lo, hi) or a list/tuple of (lo, hi) ranges.
               nested = (
                   len(band) > 0 and isinstance(band[0], (list, tuple, np.ndarray))
               )
               band_spans = [tuple(b) for b in band] if nested else [tuple(band)]
           for j, span in enumerate(band_spans):
               lo, hi = float(span[0]), float(span[1])
               if not (np.isfinite(lo) and np.isfinite(hi)):
                   continue
               ax.axvspan(
                   lo, hi, color=band_color, alpha=band_alpha, zorder=0,
                   label=band_label if (band_label and j == 0) else None,
               )

       # Build layer -> mean physical distance lookup from target stats
       layer_to_dist = {}
       for stats_df in target_grouped_stats.values():
           if "distance" not in stats_df.columns:
               continue
           for _, row in stats_df.iterrows():
               lyr = int(round(row["layer"]))
               dist = row["distance"]
               if dist is not None and not (isinstance(dist, float) and np.isnan(dist)):
                   layer_to_dist.setdefault(lyr, []).append(dist)
       layer_to_dist = {lyr: float(np.mean(vals)) for lyr, vals in layer_to_dist.items()}

       if layer_to_dist and unit:
           # Bottom axis (ax): relabel ticks with physical distance values
           def phys_formatter(value, _pos):
               lyr = int(round(value))
               return f"{layer_to_dist[lyr]:.1f}" if lyr in layer_to_dist else ""
           ax.xaxis.set_major_formatter(FuncFormatter(phys_formatter))
           ax.set_xlabel(f"Physical distance from {display_base_type} border ({unit})")

           # Top axis (ax2 via twiny): cellular layer index ticks
           ax2 = ax.twiny()
           ax2.set_xlim(ax.get_xlim())
           primary_ticks = [t for t in ax.get_xticks() if int(round(t)) in layer_to_dist]
           ax2.set_xticks(primary_ticks)
           ax2.set_xticklabels([f"{int(round(t))}" for t in primary_ticks])
           ax2.set_xlabel(f"Cellular distance from {display_base_type} border (layers)")
           ax2.tick_params(axis="x")
       else:
           ax.ticklabel_format(axis="x", style="plain", useOffset=False)
           ax.set_xlabel(f"Cellular distance from {display_base_type} border (layers)")

       # Optional per-layer p-value track on a secondary log y-axis.
       pvalue_handle = None
       if pvalue_show:
           if pvalue_stats is None or len(pvalue_stats) == 0:
               raise ValueError("pvalue_show=True but no pvalue_stats provided.")
           pcol = "p_adj" if pvalue_use_adjusted else "p_value"
           if pcol not in pvalue_stats.columns:
               raise ValueError(f"pvalue_stats missing '{pcol}' column.")
           pstats = pvalue_stats.sort_values("layer")
           xp = pstats["layer"].round().astype(np.int32).to_numpy()
           yp = pstats[pcol].to_numpy(dtype=float)
           finite = np.isfinite(yp)
       # Only build the secondary p-value axis when at least one layer has a
       # valid p-value. With no testable layer there is nothing to plot, so an
       # empty twin axis (bare threshold line + label) would be misleading.
       if pvalue_show and finite.any():
           axp = ax.twinx()
           axp.set_yscale("log")
           # Fix the y-range up front so the p=threshold reference is always
           # in-frame: extend the bottom past both the smallest observed p and
           # the threshold. This keeps the threshold meaningful even when every
           # layer is non-significant (whole curve above 0.05) without wasting
           # log-resolution when p's get tiny.
           y_top = 1.0
           y_bottom = np.nanmin(yp[finite]) * 0.5
           if pvalue_threshold_show and pvalue_threshold is not None:
               y_bottom = min(y_bottom, pvalue_threshold * 0.5)
           y_bottom = max(y_bottom, 1e-12)
           if pvalue_ylim is not None:
               # explicit fixed range (bottom, top) -- overrides the per-panel
               # auto-scaling so p-axes are comparable across panels.
               y_bottom, y_top = pvalue_ylim
           axp.set_ylim(top=y_top, bottom=y_bottom)
           (pvalue_handle,) = axp.plot(
               xp[finite],
               yp[finite],
               color=pvalue_color,
               linestyle="--",
               linewidth=1.2,
               marker=None,
               label=pvalue_label,
           )
           # Only draw the threshold line/label when it lies within the axis.
           if (
               pvalue_threshold_show
               and pvalue_threshold is not None
               and y_bottom <= pvalue_threshold <= y_top
           ):
               axp.axhline(
                   pvalue_threshold,
                   color="0.35",
                   linestyle=(0, (1, 1)),
                   linewidth=0.8,
                   alpha=1.0,
                   zorder=5,
               )
               axp.text(
                   0.995,
                   pvalue_threshold,
                   f"p = {pvalue_threshold:g}",
                   transform=axp.get_yaxis_transform(),
                   ha="right",
                   va="bottom",
                   color="0.35",
                   fontsize=8,
                   alpha=1.0,
                   clip_on=True,
               )
           axp.set_ylabel(pvalue_label)
           axp.grid(False)

       # Legend ordering
       handles, labels = ax.get_legend_handles_labels()
       if legend_order is not None:
           idx = [labels.index(l) for l in legend_order if l in labels]
           handles = [handles[i] for i in idx]
           labels = [labels[i] for i in idx]
       if pvalue_handle is not None:
           handles = list(handles) + [pvalue_handle]
           labels = list(labels) + [pvalue_label]
       ax.legend(handles, labels, title=legend_title, **legend_kwargs)
   return ax


def plot_hplot_gam(
    grid,
    curves,
    *,
    ax=None,
    group_labels=None,
    color_map=None,
    palette=None,
    ci_show=True,
    ci_alpha=0.18,
    linewidth=2.0,
    zero_line=True,
    ref_band=None,
    ref_peak=None,
    ref_band_color="0.6",
    ref_band_alpha=0.12,
    ref_peak_color="0.3",
    xlabel="border layer L",
    ylabel="value",
    xlim=None,
    legend=True,
    legend_fontsize=8,
    legend_loc="upper left",
):
    """Draw a GAM-smoothed H-plot (**H-GAM Plot**) panel from per-group smooths.

    Renders the penalised-spline smooth of each group with its pointwise CI
    band, i.e. the top panel that pairs with :func:`plot_hplot_gam_delta`. This
    is the lightweight, grid-friendly functional counterpart to
    ``HPlot.fit(smoother="gam").plot()`` — it draws only the smooth curves (no
    raw layer means, no secondary distance axis) so it composes cleanly into a
    dense multi-panel figure.

    Parameters
    ----------
    grid : array-like, shape (n_grid,)
        Layer coordinates that the prediction arrays are evaluated on.
    curves : dict
        Output of :func:`hplot.stats.gam_group_curves`:
        ``{group_label: (pred_array, ci_array)}`` with *pred_array* shape
        ``(n_grid,)`` and *ci_array* shape ``(n_grid, 2)`` (columns
        ``[lower, upper]``).
    ax : matplotlib.axes.Axes | None
        Axis to draw into; a new one is created when ``None``.
    group_labels : sequence | None
        Order (and optional subset) of ``curves`` keys to draw. Defaults to the
        insertion order of ``curves``.
    color_map : dict | None
        Mapping ``group_label -> colour``. Groups not present fall back to
        ``palette``.
    palette : sequence | None
        Colour cycle used when a label is absent from ``color_map``. Defaults to
        ``matplotlib.cm.tab10``.
    ci_show : bool
        Shade the pointwise CI band. Default ``True``.
    ci_alpha : float
        Opacity of the CI shading. Default ``0.18``.
    linewidth : float
        Width of the smooth line. Default ``2.0``.
    zero_line : bool
        Draw a dashed vertical reference at layer 0 (the boundary). Default
        ``True``.
    ref_band : tuple[float, float] | None
        ``(lo, hi)`` reference span (e.g. a Stage-1 cluster-mass band).
    ref_peak : float | None
        Reference x-position (e.g. a Stage-1 peak layer).
    ref_band_color, ref_band_alpha, ref_peak_color : str, float, str
        Styling of the reference markers.
    xlabel, ylabel : str
        Axis labels. Pass ``""`` to leave an axis unlabelled (useful for inner
        panels of a grid).
    xlim : tuple[float, float] | None
        Explicit x-range; defaults to ``(grid[0], grid[-1])``.
    legend : bool
        Draw a legend of the group labels. Default ``True``.
    legend_fontsize : float
        Font size for the legend.
    legend_loc : str
        Legend location.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into.
    """
    grid = np.asarray(grid, dtype=float)
    if group_labels is None:
        group_labels = list(curves.keys())
    if color_map is None and palette is None:
        palette = plt.cm.tab10.colors

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.2))

    for i, label in enumerate(group_labels):
        pred, ci = curves[label]
        pred = np.asarray(pred, dtype=float)
        if color_map is not None and label in color_map:
            color = color_map[label]
        else:
            color = palette[i % len(palette)]
        ax.plot(grid, pred, color=color, lw=linewidth, label=str(label))
        if ci_show and ci is not None:
            ci = np.asarray(ci, dtype=float)
            ax.fill_between(grid, ci[:, 0], ci[:, 1], color=color, alpha=ci_alpha, lw=0)

    if ref_band is not None:
        ax.axvspan(
            float(ref_band[0]), float(ref_band[1]),
            color=ref_band_color, alpha=ref_band_alpha, zorder=0,
        )
    if ref_peak is not None:
        ax.axvline(float(ref_peak), color=ref_peak_color, lw=0.8, ls=":", alpha=0.7)
    if zero_line:
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)

    ax.set_xlim(xlim if xlim is not None else (grid[0], grid[-1]))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    return ax


def plot_hplot_gam_delta(
    grid,
    diff_pred,
    ci_lower,
    ci_upper,
    sig_pos,
    sig_neg,
    *,
    ax=None,
    group_labels=("low", "high"),
    high_color="#d62728",
    low_color="#1f77b4",
    line_color="0.3",
    ci_color="0.75",
    ci_alpha=0.35,
    sig_alpha=0.45,
    ref_band=None,
    ref_peak=None,
    ref_band_color="0.6",
    ref_band_alpha=0.10,
    ref_peak_color="0.3",
    xlabel="border layer L",
    ylabel="\u0394 (high \u2212 low)",
    xlim=None,
    show_sig_pct=True,
    show_legend=True,
    legend_fontsize=6,
    boundary_line=True,
    boundary_line_color="black",
    boundary_line_width=1.0,
    boundary_line_alpha=0.5,
):
    """Draw a differential H-GAM (**H-ΔGAM Plot**) panel.

    Renders the layer-wise difference curve
    :math:`\\Delta(L) = f_{high}(L) - f_{low}(L)` produced by
    :func:`hplot.stats.gam_delta_curve`: a propagated 95 % CI band that is
    coloured with ``high_color`` where the high group is pointwise larger
    (``sig_pos``) and ``low_color`` where the low group is pointwise larger
    (``sig_neg``), grey elsewhere, with a zero reference line.

    This is the companion of the H-GAM Plot top panel (the per-group smooths,
    drawn by :func:`plot_hplot` with ``smoother='gam'`` / ``gam_curves=``).

    .. warning::
        The ``sig_pos`` / ``sig_neg`` colouring reflects **pointwise,
        multiple-comparison-uncorrected** CI exclusion of zero (see the warning
        in :func:`hplot.stats.gam_delta_curve`). Treat it as
        visualisation / localisation only. For confirmatory inference use
        :func:`hplot.stats.cluster_mass_screen` with FDR control. The optional
        ``ref_band`` / ``ref_peak`` markers come from that *separate* method
        (e.g. a Stage-1 cluster-mass band) and are drawn only as a light
        reference; they are not implied to coincide with the coloured region.

    Parameters
    ----------
    grid : array-like, shape (n_grid,)
        Layer coordinates aligned with the difference arrays (the GAM grid).
    diff_pred, ci_lower, ci_upper : array-like, shape (n_grid,)
        Point estimate and propagated CI bounds from
        :func:`hplot.stats.gam_delta_curve`.
    sig_pos, sig_neg : array-like of bool, shape (n_grid,)
        Pointwise significance masks (high>low and low>high respectively).
    ax : matplotlib.axes.Axes | None
        Axis to draw into; a new one is created when ``None``.
    group_labels : tuple[str, str]
        ``(low_label, high_label)`` used to build the legend entries.
    high_color, low_color : str
        Fill colours for the ``sig_pos`` and ``sig_neg`` regions.
    line_color : str
        Colour of the :math:`\\Delta` point-estimate line.
    ci_color, ci_alpha : str, float
        Colour / opacity of the full (non-significant) CI band.
    sig_alpha : float
        Opacity of the coloured significant regions.
    ref_band : tuple[float, float] | None
        ``(lo, hi)`` reference span (e.g. a Stage-1 cluster-mass band).
    ref_peak : float | None
        Reference x-position (e.g. a Stage-1 peak layer).
    ref_band_color, ref_band_alpha, ref_peak_color : str, float, str
        Styling of the reference markers.
    xlabel, ylabel : str
        Axis labels.
    xlim : tuple[float, float] | None
        Explicit x-range; defaults to ``(grid[0], grid[-1])``.
    show_sig_pct : bool
        Annotate the fraction of the layer range that is pointwise significant.
    show_legend : bool
        Draw a legend when any region is significant.
    legend_fontsize : float
        Font size for the legend.
    boundary_line : bool
        Draw a dashed **vertical** reference at layer 0 (the boundary),
        mirroring the ``zero_line`` of the H-GAM Plot top panel so the two
        panels align at the boundary. This is distinct from the horizontal
        :math:`\\Delta = 0` line, which is always drawn. Default ``True``.
    boundary_line_color, boundary_line_width, boundary_line_alpha : str, float, float
        Styling of the vertical boundary line.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into.
    """
    grid = np.asarray(grid, dtype=float)
    diff_pred = np.asarray(diff_pred, dtype=float)
    ci_lower = np.asarray(ci_lower, dtype=float)
    ci_upper = np.asarray(ci_upper, dtype=float)
    sig_pos = np.asarray(sig_pos, dtype=bool)
    sig_neg = np.asarray(sig_neg, dtype=bool)
    lo_lab, hi_lab = group_labels

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 1.8))

    ax.fill_between(grid, ci_lower, ci_upper, color=ci_color, alpha=ci_alpha, lw=0)
    ax.fill_between(
        grid, ci_lower, ci_upper, where=sig_pos,
        color=high_color, alpha=sig_alpha, lw=0, label=f"{hi_lab}>{lo_lab}",
    )
    ax.fill_between(
        grid, ci_lower, ci_upper, where=sig_neg,
        color=low_color, alpha=sig_alpha, lw=0, label=f"{lo_lab}>{hi_lab}",
    )
    ax.plot(grid, diff_pred, color=line_color, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    if boundary_line:
        ax.axvline(
            0, color=boundary_line_color, lw=boundary_line_width,
            ls="--", alpha=boundary_line_alpha,
        )

    if ref_band is not None:
        ax.axvspan(
            float(ref_band[0]), float(ref_band[1]),
            color=ref_band_color, alpha=ref_band_alpha, zorder=0,
        )
    if ref_peak is not None:
        ax.axvline(float(ref_peak), color=ref_peak_color, lw=0.8, ls=":", alpha=0.7)

    ax.set_xlim(xlim if xlim is not None else (grid[0], grid[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_sig_pct and len(grid):
        pct = 100.0 * int(np.sum(sig_pos) + np.sum(sig_neg)) / len(grid)
        ax.text(
            0.97, 0.90, f"{pct:.0f}% sig.", ha="right", va="top",
            transform=ax.transAxes, fontsize=6.5,
        )
    if show_legend and (np.any(sig_pos) or np.any(sig_neg)):
        ax.legend(
            fontsize=legend_fontsize, loc="lower left",
            handlelength=1, handletextpad=0.4, borderpad=0.3,
        )
    return ax


def plot_hplotx(grouped_stats, unit=None, ci_show=True, ax=None, display_base_type='tumor', display_target_type='immune cells', value_kind="proportion", ylabel=None, color_map=None, palette=None, legend_order=None, legend_title="Group", legend_kwargs=None,):

    if color_map is None and palette is None:
        palette = plt.cm.tab10.colors
        
    labels = list(grouped_stats.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    plt.tight_layout()
    
    for label, df in grouped_stats.items():
        x = df['layer'].round().astype(np.int32)
        y = df["mean"]

        if color_map is not None:
            color = color_map[label]
        else:
            color = palette[i % len(palette)]

        ax.plot(x, y, label=str(label), color=color)

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], color=color, alpha=0.3)

    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

    def distance_formattyer(val, pos):
        dst_list = []
        for _, df in grouped_stats.items():
            if int(round(val)) in df['layer'].round().astype(np.int32).tolist():
                dst = df[df['layer'].round().astype(np.int32)==int(round(val))]['distance'].mean()
                dst_list.append(dst)
        
        if len(dst_list) > 0:
            dst_mean = np.mean(dst_list)
            tick_label = f"{val:g}\n({dst_mean:.1f})"
        else:
            tick_label = f"{val:g}\n"
            
        return tick_label

    ax.xaxis.set_major_formatter(FuncFormatter(distance_formattyer))
    ax.set_xlabel(f"Layerwise cellular distance from {display_base_type} border\n(Physical distance{' ('+unit+') ' if unit else ' '}from {display_base_type} border)")  
    ax.set_ylabel(_build_ylabel(value_kind, display_target_type, ylabel))
    ax.set_title("Tumor Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax


def _hloci_up_mask(directions):
    """Coerce a per-row direction spec into a boolean ``is_up`` mask.

    Accepts a boolean array (``True`` = up), a numeric array (``> 0`` = up), or
    a string array (tokens such as ``"elevated"``/``"up"``/``"high"``/``"+"``
    are up; everything else is down).
    """
    arr = np.asarray(directions)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.number):
        return arr > 0
    up_tokens = {"elevated", "up", "high", "pos", "positive", "+", "1", "true"}
    return np.array([str(v).strip().lower() in up_tokens for v in arr], dtype=bool)


def _hloci_dir_code(directions):
    """Coerce a per-row direction spec into an int code: ``+1`` up, ``-1`` down,
    ``0`` neutral.

    Numeric input uses the sign of the value, with ``0`` and ``NaN`` mapping to
    neutral (``0``) — so a group comparison with no signed direction (e.g. a
    4-arm Kruskal–Wallis screen, where every row is ``0``/``NaN``) renders in
    the neutral ``default_color`` rather than being forced up or down. Boolean
    input stays two-state (``True`` -> up, ``False`` -> down; no neutral), which
    preserves the behaviour of callers that pass a side mask. String tokens map
    ``elevated``/``up``/… -> up, ``depressed``/``down``/… -> down, and blank or
    unrecognised tokens -> neutral.
    """
    arr = np.asarray(directions)
    if arr.dtype == bool:
        return np.where(arr, 1, -1).astype(int)
    if np.issubdtype(arr.dtype, np.number):
        return np.sign(np.nan_to_num(arr.astype(float), nan=0.0)).astype(int)
    up_tokens = {"elevated", "up", "high", "pos", "positive", "+", "1", "true"}
    down_tokens = {"depressed", "down", "low", "neg", "negative", "-"}
    codes = []
    for v in arr:
        s = str(v).strip().lower()
        codes.append(1 if s in up_tokens else (-1 if s in down_tokens else 0))
    return np.asarray(codes, dtype=int)


def plot_hloci_strip(
    positions,
    directions,
    *,
    weights=None,
    labels=None,
    ax=None,
    strip_color="k",
    lw_range=(1.0, 12.0),
    default_lw=2.0,
    half_height=0.30,
    marker_size=6.0,
    gap_pt=0.9,
    up_marker="^",
    down_marker="v",
    marker_edge_width=0.4,
    zero_line=True,
    zero_line_color="0.45",
    zero_line_width=0.9,
    grid_axis="x",
    xlabel="signed cluster mass",
    ylabel=None,
    title=None,
    title_fontsize=None,
):
    """Draw an **H-Loci Summary**: a location-of-the-border-band panel.

    Each row is one feature's **H-Locus** — its cluster-mass profile localised
    over the spatial domain of the tissue microenvironment — drawn as a short
    vertical **strip** planted at its position ``x`` along the border axis. The
    strip's *thickness* encodes a magnitude (``weights``, e.g. cluster mass) and
    a **triangle** glyph encodes polarity (``directions``): ``▲`` above the
    strip for the *up* group (e.g. elevated), ``▼`` below for the *down* group
    (e.g. depressed). Stacking many rows gives the **H-Loci Summary** — an
    integrated view of many features' loci across the microenvironment — with
    three channels read at a glance: horizontal position (localisation), strip
    thickness (effect magnitude / cluster mass), and triangle direction (sign of
    the effect).

    This is the packaged, reusable form of the per-feature "Location of the
    border band" panel used alongside a significance bar chart. The triangle is
    offset from the strip end by a fixed number of *points* (independent of the
    number of rows and of the export dpi), so the gap between strip and glyph
    stays visually constant and the glyph stays exactly centred on the strip
    (no horizontal shift).

    Parameters
    ----------
    positions : array-like, shape (n,)
        Signed x-position of each row (e.g. signed cluster mass; sign encodes
        the side of the border, magnitude the location).
    directions : array-like, shape (n,)
        Per-row polarity controlling the triangle. Boolean (``True`` = up),
        numeric (``> 0`` = up), or string (``"elevated"``/``"up"``/``"high"``…
        = up). See :func:`_hloci_up_mask`.
    weights : array-like, shape (n,) | None
        Non-negative magnitude mapped to strip linewidth via ``lw_range``
        (min→max). When ``None`` every strip uses ``default_lw``.
    labels : sequence | None
        Row (y-tick) labels, bottom to top. When ``None`` no y-tick labels.
    ax : matplotlib.axes.Axes | None
        Axis to draw into; a new one is created when ``None``.
    strip_color : str
        Colour of the strips and triangles. Default ``"k"``.
    lw_range : tuple[float, float]
        ``(min_lw, max_lw)`` linewidth span mapped from ``weights``. Default
        ``(1.0, 12.0)`` for a wide dynamic range; the smallest weight maps to
        ``min_lw`` and the largest to ``max_lw``.
    default_lw : float
        Strip linewidth used when ``weights`` is ``None`` or constant.
    half_height : float
        Half-height of each strip in row-index units. Default ``0.30``.
    marker_size : float
        Triangle marker size in points. Default ``6.0``.
    gap_pt : float
        Gap between the strip end and the triangle, in points (dpi-independent).
        Default ``0.9`` (≈ 3 px at 240 dpi).
    up_marker, down_marker : str
        Matplotlib markers for the up / down groups. Default ``"^"`` / ``"v"``.
    marker_edge_width : float
        Triangle edge linewidth.
    zero_line : bool
        Draw a vertical reference line at ``x = 0`` (the boundary).
    zero_line_color, zero_line_width : str, float
        Styling of the zero reference line.
    grid_axis : str | None
        Axis for a light background grid (``"x"``, ``"y"``, ``"both"`` or
        ``None`` to disable). Default ``"x"``.
    xlabel, ylabel, title : str | None
        Axis labels / title. Pass ``None`` (or ``""`` for the axis labels) to
        leave unset.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into.
    """
    from matplotlib.transforms import offset_copy

    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    is_up = _hloci_up_mask(directions)
    if len(is_up) != n:
        raise ValueError("positions and directions must have the same length.")

    if weights is None:
        lw = np.full(n, default_lw, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("weights must match the length of positions.")
        lw = (np.interp(w, (w.min(), w.max()), lw_range)
              if n and w.max() > w.min() else np.full(n, default_lw, dtype=float))

    if ax is None:
        fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 18.0))
        _, ax = plt.subplots(figsize=(5.8, fig_h))
    fig = ax.figure

    y = np.arange(n)
    # triangle CENTRE offset beyond the strip end, in points (dpi-independent);
    # x-offset is 0 so the glyph stays exactly centred on the strip.
    off = marker_size / 2.0 + gap_pt
    tr_up = offset_copy(ax.transData, fig=fig, x=0.0, y=+off, units="points")
    tr_down = offset_copy(ax.transData, fig=fig, x=0.0, y=-off, units="points")

    for _yi, _x, _lwi, _up in zip(y, pos, lw, is_up):
        ax.plot([_x, _x], [_yi - half_height, _yi + half_height],
                color=strip_color, lw=_lwi, solid_capstyle="butt", zorder=3)
        if _up:
            ax.plot([_x], [_yi + half_height], marker=up_marker, color=strip_color,
                    ms=marker_size, mec=strip_color, mew=marker_edge_width,
                    zorder=4, transform=tr_up, clip_on=False)
        else:
            ax.plot([_x], [_yi - half_height], marker=down_marker, color=strip_color,
                    ms=marker_size, mec=strip_color, mew=marker_edge_width,
                    zorder=4, transform=tr_down, clip_on=False)

    if zero_line:
        ax.axvline(0.0, color=zero_line_color, lw=zero_line_width, zorder=1)

    ax.set_yticks(y)
    if labels is not None:
        ax.set_yticklabels(list(labels), fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.9", lw=0.6)
    return ax


def _alpha_ramp_cmap(color, a_lo, a_hi, n=256):
    """Build a colormap of constant hue ``color`` whose alpha ramps ``a_lo``→``a_hi``.

    Used to render a mass colorbar for the H-Loci Summary, where cluster mass
    is encoded by opacity (alpha) at a fixed direction colour.
    """
    from matplotlib.colors import to_rgb, ListedColormap
    r, g, b = to_rgb(color)
    a = np.linspace(float(a_lo), float(a_hi), int(n))
    return ListedColormap(
        np.column_stack([np.full(n, r), np.full(n, g), np.full(n, b), a]))


def plot_hloci_bands(
    band_lo,
    band_hi,
    directions,
    *,
    peak=None,
    mass=None,
    labels=None,
    sort="outer_to_inner",
    ax=None,
    up_color="#d62728",
    down_color="#1f77b4",
    default_color="0.6",
    bar_height=0.6,
    alpha=0.85,
    alpha_range=None,
    edge_color="none",
    edge_width=0.0,
    peak_marker=True,
    peak_marker_color="0.15",
    peak_marker_width=1.4,
    boundary_line=True,
    boundary_line_color="0.5",
    boundary_line_width=0.9,
    grid_axis="x",
    xlabel="border layer L",
    ylabel=None,
    title=None,
    title_fontsize=None,
    mass_colorbar=False,
    mass_colorbar_label="cluster mass",
    mass_colorbar_loc="lower right",
    mass_colorbar_size=(1.5, 0.09),
    mass_colorbar_pad=0.8,
    mass_colorbar_bbox=None,
    mass_colorbar_tags=True,
):
    """Draw an **H-Loci Summary (band mode)** panel from cluster-band extents.

    Alternative rendering of the H-Loci Summary in which each row is one
    feature's **H-Locus** drawn as a *horizontal bar* spanning its actual
    cluster band ``[band_lo, band_hi]`` on the signed border-layer axis, filled
    by direction (``up_color`` for elevated, ``down_color`` for depressed). This
    is the bar-chart counterpart of the direction-shaded ``axvspan`` band used
    on the per-feature H-plots (i.e. it shows *where* the band sits and *how
    wide* it is — the cluster width — rather than mapping the mass to a strip
    thickness as :func:`plot_hloci_strip` does).

    Parameters
    ----------
    band_lo, band_hi : array-like, shape (n,)
        Lower / upper layer bounds of each feature's scored cluster band. Rows
        with a non-finite bound are skipped.
    directions : array-like, shape (n,)
        Per-feature direction, coerced by :func:`_hloci_dir_code` into a
        three-state code. Numeric input uses the sign of the value: ``> 0`` rows
        are filled with ``up_color``, ``< 0`` rows with ``down_color``, and
        ``0``/``NaN`` rows with ``default_color`` (neutral) — so a group screen
        with no signed direction (e.g. a 4-arm Kruskal–Wallis screen) renders
        neutral instead of being forced up or down. Boolean input stays
        two-state (``True`` -> up, ``False`` -> down); string tokens map
        ``elevated``/``up`` -> up, ``depressed``/``down`` -> down, else neutral.
    peak : array-like, shape (n,) | None
        Optional per-feature peak layer; when given a short vertical tick is
        drawn inside each bar at that layer.
    mass : array-like, shape (n,) | None
        Optional per-feature cluster mass, used **only** as the secondary sort
        key (see ``sort``). Larger mass ranks first among rows that share the
        same primary sort position.
    labels : sequence | None
        Row (y-tick) labels.
    sort : {"outer_to_inner", "inner_to_outer"} | None
        Default row ordering, keyed on the **peak centre of cluster mass**
        (``peak`` when supplied, else the band midpoint). ``"outer_to_inner"``
        (default) places the outermost H-Loci (stroma side, high layer L) at
        the top and the innermost (tumour interior, low L) at the bottom;
        ``"inner_to_outer"`` reverses this. Ties on the primary key are broken
        by descending ``mass`` (cluster mass) when supplied, else by the
        caller-supplied row order. Pass ``None`` (or ``False``) to keep the
        caller-supplied row order.
    ax : matplotlib.axes.Axes | None
        Axis to draw into; a new one is created when ``None``.
    up_color, down_color : str
        Fill colours for elevated / depressed bands.
    default_color : str
        Fallback fill when a direction cannot be resolved.
    bar_height : float
        Height of each horizontal bar (in data units; rows are unit-spaced).
    alpha : float
        Bar fill opacity when ``alpha_range`` is ``None`` (constant for every
        bar).
    alpha_range : tuple[float, float] | None
        When given together with ``mass``, the per-bar opacity encodes the
        cluster mass: the smallest mass maps to ``alpha_range[0]`` (most
        transparent) and the largest to ``alpha_range[1]`` (most opaque), so
        transparency becomes a magnitude channel. ``None`` (default) uses the
        constant ``alpha`` for all bars.
    edge_color : str
        Bar edge colour. Default ``"none"``.
    edge_width : float
        Bar edge width.
    alpha_range : tuple[float, float] | None
        When given together with ``mass``, per-bar opacity encodes the cluster
        mass: smallest mass -> ``alpha_range[0]`` (most transparent), largest ->
        ``alpha_range[1]`` (most opaque). ``None`` uses the constant ``alpha``.
    peak_marker : bool
        Draw the per-feature peak tick when ``peak`` is provided.
    peak_marker_color, peak_marker_width : str, float
        Styling of the peak tick.
    boundary_line : bool
        Draw a dashed **vertical** reference at layer 0 (the boundary).
    boundary_line_color, boundary_line_width : str, float
        Styling of the boundary line.
    grid_axis : str | None
        Axis for a light grid (``"x"``, ``"y"``, ``"both"`` or ``None``).
    xlabel, ylabel, title : str | None
        Axis labels / title.
    title_fontsize : float | None
        Font size for the title.
    mass_colorbar : bool
        When ``True`` (and both ``mass`` and ``alpha_range`` are supplied), add
        a slim colorbar for each direction present (``up_color`` for elevated,
        ``down_color`` for depressed). Each colorbar shows the same fixed hue
        with alpha ramping ``alpha_range[0]``→``alpha_range[1]`` across the
        observed cluster-mass range, i.e. it is the legend for the opacity =
        cluster-mass encoding.
    mass_colorbar_label : str
        Base label for the mass colorbar(s); the direction name is appended.
    mass_colorbar_loc : str
        Legend-style corner anchor for the colorbar block (e.g. ``"lower
        right"``, ``"upper left"``). Used when ``mass_colorbar_bbox`` is
        ``None``. The block is placed at a **fixed physical size** and stays
        anchored to that corner regardless of the panel height, exactly like
        :meth:`~matplotlib.axes.Axes.legend`.
    mass_colorbar_size : tuple[float, float]
        ``(width_in, strip_height_in)`` in **inches** for the colorbar block:
        the overall width and the height of each single direction strip. Fixed
        physical size is what keeps the block legend-sized on a very tall panel.
    mass_colorbar_pad : float
        Border padding (in font-size units, like a legend ``borderpad``)
        between the colorbar block and the anchored corner.
    mass_colorbar_bbox : tuple[float, float, float, float] | None
        Legacy override: ``(x0, y0, width, height)`` in **axes fraction** for
        explicit placement of the first strip (additional strips stack below).
        When ``None`` (default) the legend-style ``mass_colorbar_loc`` /
        ``mass_colorbar_size`` placement is used instead.
    mass_colorbar_tags : bool
        When ``True`` (default) write the direction name (``elevated`` /
        ``depressed``) to the left of each strip, so the colorbar doubles as
        the direction legend. Disable when the strips sit near the y-tick
        labels (e.g. an upper-left placement) to avoid overlap.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into.
    """
    lo = np.asarray(band_lo, dtype=float)
    hi = np.asarray(band_hi, dtype=float)
    n = len(lo)
    if len(hi) != n:
        raise ValueError("band_lo and band_hi must have the same length.")
    is_up = _hloci_dir_code(directions)
    if len(is_up) != n:
        raise ValueError("directions must match the length of band_lo.")
    pk = np.asarray(peak, dtype=float) if peak is not None else None
    if pk is not None and len(pk) != n:
        raise ValueError("peak must match the length of band_lo.")
    ms = np.asarray(mass, dtype=float) if mass is not None else None
    if ms is not None and len(ms) != n:
        raise ValueError("mass must match the length of band_lo.")

    # Default row ordering by the peak centre of cluster mass. barh draws row 0
    # at the bottom and row n-1 at the top, so an ascending sort of the centre
    # puts the innermost H-Locus at the bottom and the outermost at the top —
    # i.e. reading the panel top-to-bottom goes outer -> inner. Ties on the
    # centre are broken by descending cluster mass (deterministic two-key
    # sort) when `mass` is supplied.
    if sort:
        _center = pk.copy() if pk is not None else (lo + hi) / 2.0
        _center = np.where(np.isfinite(_center), _center, -np.inf)
        _primary = -_center if str(sort).replace("-", "_") == "inner_to_outer" \
            else _center
        if ms is not None:
            _sec = np.where(np.isfinite(ms), ms, -np.inf)
            # lexsort: last key is primary; -_sec ascending => mass descending.
            _order = np.lexsort((-_sec, _primary))
        else:
            _order = np.argsort(_primary, kind="stable")
        lo, hi, is_up = lo[_order], hi[_order], is_up[_order]
        if pk is not None:
            pk = pk[_order]
        if ms is not None:
            ms = ms[_order]
        if labels is not None:
            labels = [list(labels)[i] for i in _order]

    if ax is None:
        fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 18.0))
        _, ax = plt.subplots(figsize=(5.8, fig_h))

    # Per-bar opacity: encode cluster mass through alpha when requested,
    # otherwise use the constant `alpha` for every bar.
    if alpha_range is not None and ms is not None:
        _a = np.full(n, float(alpha_range[0]), dtype=float)
        _fin = np.isfinite(ms)
        if _fin.any():
            _mn, _mx = float(np.min(ms[_fin])), float(np.max(ms[_fin]))
            if _mx > _mn:
                _a[_fin] = np.interp(ms[_fin], (_mn, _mx), alpha_range)
            else:
                _a[_fin] = float(alpha_range[1])
    else:
        _a = np.full(n, float(alpha), dtype=float)

    y = np.arange(n)
    for _yi, _lo, _hi, _up, _ai in zip(y, lo, hi, is_up, _a):
        if not (np.isfinite(_lo) and np.isfinite(_hi)):
            continue
        _left, _width = min(_lo, _hi), abs(_hi - _lo)
        _col = up_color if _up > 0 else (down_color if _up < 0 else default_color)
        ax.barh(_yi, _width, left=_left, height=bar_height,
                color=_col, alpha=_ai, edgecolor=edge_color,
                linewidth=edge_width, zorder=3)
    if pk is not None and peak_marker:
        for _yi, _pk in zip(y, pk):
            if np.isfinite(_pk):
                ax.plot([_pk, _pk], [_yi - bar_height / 2.0, _yi + bar_height / 2.0],
                        color=peak_marker_color, lw=peak_marker_width, zorder=4)

    if boundary_line:
        ax.axvline(0.0, color=boundary_line_color, lw=boundary_line_width,
                   ls="--", zorder=1)

    ax.set_yticks(y)
    if labels is not None:
        ax.set_yticklabels(list(labels), fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.9", lw=0.6)

    # Mass colorbar: legend for the opacity = cluster-mass encoding. A single
    # compact block — one thin strip per direction present (elevated red above
    # depressed blue), sharing one mass tick-axis and one label; each strip is a
    # fixed-hue alpha ramp over the observed mass range.
    if mass_colorbar and ms is not None and alpha_range is not None:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        _fin = np.isfinite(ms)
        if _fin.any():
            _mn, _mx = float(np.min(ms[_fin])), float(np.max(ms[_fin]))
            if _mx <= _mn:
                _mx = _mn + 1.0
            _norm = Normalize(vmin=_mn, vmax=_mx)
            _dirs = []
            if np.any(is_up > 0):
                _dirs.append(("elevated", up_color))
            if np.any(is_up < 0):
                _dirs.append(("depressed", down_color))
            if np.any(is_up == 0):
                _dirs.append(("neutral", default_color))

            def _draw_strip(_cax, _name, _col, _show_ticks):
                _cmap = _alpha_ramp_cmap(_col, alpha_range[0], alpha_range[1])
                _sm = ScalarMappable(norm=_norm, cmap=_cmap)
                _sm.set_array([])
                _cb = ax.figure.colorbar(_sm, cax=_cax, orientation="horizontal")
                _cb.ax.xaxis.set_ticks_position("bottom")
                if _show_ticks:
                    _cb.set_label(mass_colorbar_label, fontsize=7, labelpad=2)
                    _cb.ax.tick_params(labelsize=6)
                else:
                    _cb.ax.set_xticklabels([])
                    _cb.ax.tick_params(length=0)
                if mass_colorbar_tags:
                    _cb.ax.text(-0.04, 0.5, _name, transform=_cb.ax.transAxes,
                                ha="right", va="center", fontsize=6, color=_col)

            if mass_colorbar_bbox is not None:
                # legacy: explicit axes-fraction placement, strips stacked down.
                _bx, _by, _bw, _bh = mass_colorbar_bbox
                for _i, (_name, _col) in enumerate(_dirs):
                    _cax = ax.inset_axes(
                        [_bx, _by - _i * (_bh * 1.25), _bw, _bh])
                    _draw_strip(_cax, _name, _col, _i == len(_dirs) - 1)
            else:
                # legend-style placement: a fixed-inch block anchored to a
                # corner (like Axes.legend), so it never drifts or overlaps
                # bars as the panel height changes. One thin strip per
                # direction, stacked inside the block.
                from mpl_toolkits.axes_grid1.inset_locator import (
                    inset_axes as _inset_axes)
                _w_in, _sh_in = mass_colorbar_size
                _n = len(_dirs)
                _gap_in = _sh_in * 0.5
                _tick_in = 0.30   # room under the bottom strip for ticks+label
                _block_in = _n * _sh_in + (_n - 1) * _gap_in + _tick_in
                _holder = _inset_axes(
                    ax, width=_w_in, height=_block_in, loc=mass_colorbar_loc,
                    borderpad=mass_colorbar_pad)
                _holder.set_axis_off()
                _sh = _sh_in / _block_in
                _gap = _gap_in / _block_in
                for _i, (_name, _col) in enumerate(_dirs):
                    _y = 1.0 - (_i + 1) * _sh - _i * _gap
                    _cax = _holder.inset_axes([0.0, _y, 1.0, _sh])
                    _draw_strip(_cax, _name, _col, _i == len(_dirs) - 1)
    return ax


def build_layer_distance_map(layers, distances=None):
    """Build a ``{layer L: mean physical distance (µm)}`` map.

    Companion to :func:`add_border_distance_axis`. The signed border layer
    ``L`` is a discrete index, but each ``L`` corresponds to a physical
    distance from the boundary that varies slightly between slides; this
    helper averages that distance per layer so the H-Loci Summary can be
    re-labelled in µm.

    Parameters
    ----------
    layers, distances : array-like
        Either two aligned flat iterables (``layers`` and ``distances``), or,
        when ``distances`` is ``None``, a single iterable of
        ``(layers, distances)`` array pairs (e.g. one pair per slide) that are
        pooled together.

    Returns
    -------
    dict[int, float]
        Map from integer layer ``L`` to its mean distance in µm.
    """
    acc = {}
    pairs = [(layers, distances)] if distances is not None else list(layers)
    for _lay, _dist in pairs:
        for _L, _d in zip(np.asarray(_lay).astype(int),
                          np.asarray(_dist, dtype=float)):
            acc.setdefault(int(_L), []).append(float(_d))
    return {L: float(np.mean(v)) for L, v in acc.items()}


def add_border_distance_axis(
    ax,
    layer_to_distance,
    *,
    max_ticks=9,
    distance_label="physical distance from border (µm)  (\u2264 0 tumour | > 0 stroma)",
    layer_label="border layer L",
    distance_fmt="{:.0f}",
    add_top_axis=True,
):
    """Re-label an H-Loci Summary band panel with physical distance (µm).

    :func:`plot_hloci_bands` draws its bars against the signed border layer
    ``L``. This helper rescales the **bottom** x-axis tick labels of that panel
    to the mean physical distance (µm) each layer corresponds to, and
    (optionally) adds a twin **top** axis that keeps the integer layer ``L``
    labels — so the same panel is readable in both units. The axis x-limits are
    preserved, so band bars stay aligned.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The band-panel axis returned by :func:`plot_hloci_bands`.
    layer_to_distance : Mapping[int, float]
        Map from integer layer ``L`` to physical distance in µm, e.g. from
        :func:`build_layer_distance_map`.
    max_ticks : int
        Maximum number of ticks to display; evenly thinned when exceeded.
    distance_label, layer_label : str
        Axis labels for the bottom (distance) and top (layer) axes. Pass an
        empty string to leave a label unchanged.
    distance_fmt : str
        Format string for the µm tick labels.
    add_top_axis : bool
        When ``True`` (default) add the twin top axis with layer labels.

    Returns
    -------
    matplotlib.axes.Axes | None
        The twin top axis when ``add_top_axis`` is ``True``, else ``None``.
    """
    xlim = ax.get_xlim()
    ticks = sorted(L for L in layer_to_distance if xlim[0] <= L <= xlim[1])
    if max_ticks and len(ticks) > max_ticks:
        ticks = ticks[:: int(np.ceil(len(ticks) / float(max_ticks)))]
    ax.set_xticks(ticks)
    ax.set_xticklabels([distance_fmt.format(layer_to_distance[L]) for L in ticks])
    if distance_label:
        ax.set_xlabel(distance_label)
    top = None
    if add_top_axis:
        top = ax.twiny()
        top.set_xlim(xlim)
        top.set_xticks(ticks)
        top.set_xticklabels([str(int(L)) for L in ticks])
        if layer_label:
            top.set_xlabel(layer_label)
    return top


def plot_hloci_fdr(
    rank,
    label_col,
    *,
    band_lo_col="band_start_layer",
    band_hi_col="band_end_layer",
    peak_col="peak_layer",
    mass_col="cluster_mass",
    fdr_col="fdr",
    direction_col=None,
    top_n=50,
    stop_label=None,
    layer_limits=None,
    layer_to_distance=None,
    fdr_threshold=0.05,
    up_color="#e6550d",
    down_color="#756bb1",
    alpha_range=(0.25, 0.9),
    analyte_label="features",
    contrast_label=None,
    n_groups=None,
    n_samples=None,
    direction_labels=None,
    title_prefix="Between-group divergence",
    title_note=None,
    ylabel=None,
    fdr_floor=None,
    width=9.6,
    legend_height=1.35,
    legend_gap=0.22,
    colorbar_y=0.30,
    fig=None,
    savepath=None,
    dpi=240,
):
    """Plot the complete H-Loci band and FDR summary from a ranking table.

    The input table is typically the ``long`` output of
    :func:`hplot.gradient_cluster_mass_screen`, augmented with feature names.
    It must contain one row per feature and the selected band extent, peak,
    cluster mass, and global FDR. The function selects features with a valid,
    non-zero band, ranks them by FDR, and creates the two-panel summary used
    to compare many H-Loci at once: coloured band spans on the left and
    ``-log10(FDR)`` on the right.

    Parameters
    ----------
    rank : pandas.DataFrame
        Feature-level H-Locus ranking table.
    label_col : str
        Column containing feature labels.
    stop_label : str | None
        Include ranked rows through this label, inclusive. If the label is not
        present among eligible rows, ``top_n`` rows are used instead.
    layer_to_distance : Mapping[int, float] | None
        Optional physical-distance map from :func:`build_layer_distance_map`.
        When supplied, the band panel receives physical-distance tick labels.
    direction_labels : sequence[tuple[str, str]] | None
        ``(name, colour)`` pairs for the cluster-mass legend, e.g.
        ``[("depressed", "#1f77b4"), ("elevated", "#d62728")]``. When ``None``
        the legend is derived from the sign of ``peak_col`` and labelled
        tumour / stroma, which only suits a position-coded screen.
    title_prefix : str
        First title line. Defaults to the between-group wording; a pooled,
        arm-free screen should pass its own.
    title_note : str | None
        Extra clause joined into the subtitle, e.g. the colour encoding.
    ylabel : str | None
        Overrides the default band-panel y-label.
    fdr_floor : float | None
        Lower clip for the FDR panel. A permutation screen cannot resolve an
        FDR below ``1 / (B + 1)``; pass that value so the bar length reports
        the test's resolution rather than the resampling budget.
    width, legend_height, legend_gap : float
        Figure width, legend-row height and the gap above it, all in inches.
        The gap is converted to a gridspec fraction so it stays constant no
        matter how many rows the panel has.
    colorbar_y : float
        Vertical position of the colour strips inside the legend row, in axes
        fraction. Raising it pulls the legend closer to the band panel by
        consuming the dead space above the strips.
    savepath : str | pathlib.Path | None
        Optional PNG output path. An SVG sibling is also written.

    Returns
    -------
    dict
        ``{"figure", "band_axis", "fdr_axis", "colorbar_axis", "selected"}``.
    """
    from pathlib import Path
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize, to_rgb

    required = {label_col, band_lo_col, band_hi_col, peak_col, mass_col, fdr_col}
    missing = required.difference(rank.columns)
    if missing:
        raise ValueError(f"rank is missing required columns: {sorted(missing)}")

    candidates = rank.loc[
        rank[peak_col].notna() & (rank[mass_col].astype(float) > 0)
    ].copy()
    n_candidates = len(candidates)
    ranked = candidates.sort_values(fdr_col, ascending=True)
    if stop_label is not None and (ranked[label_col] == stop_label).any():
        stop_index = np.flatnonzero((ranked[label_col] == stop_label).to_numpy())[0]
        selected = ranked.iloc[: stop_index + 1].copy()
    else:
        selected = ranked.head(top_n).copy()
    selected = selected.sort_values([peak_col, mass_col], ascending=[True, True]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No H-Loci with a valid peak and positive cluster mass were found.")

    n_rows = len(selected)
    fig_height = float(np.clip(0.16 * n_rows + 2.0, 4.0, 30.0))
    if fig is None:
        fig = plt.figure(figsize=(width, fig_height + legend_height + legend_gap))
    # hspace is a fraction of the mean axes height, so a fixed value would make
    # the gap grow with the row count and differ between panels of one figure set.
    grid = fig.add_gridspec(
        2, 2, width_ratios=[3, 1], height_ratios=[fig_height, legend_height],
        wspace=0.04, hspace=legend_gap / ((fig_height + legend_height) / 2.0),
    )
    band_axis = fig.add_subplot(grid[0, 0])
    fdr_axis = fig.add_subplot(grid[0, 1], sharey=band_axis)
    colorbar_axis = fig.add_subplot(grid[1, 0])
    colorbar_axis.set_axis_off()

    directions = (
        selected[direction_col].to_numpy()
        if direction_col is not None else selected[peak_col].to_numpy(dtype=float) > 0
    )
    title_parts = [str(title_prefix)]
    if contrast_label:
        title_parts.append(str(contrast_label))
    if title_note:
        title_parts.append(str(title_note))
    title_parts.append(f"top {n_rows} {analyte_label} by FDR")
    if n_groups is not None:
        title_parts.append(f"{n_groups} groups")
    if n_samples is not None:
        title_parts.append(f"pooled {n_samples} samples")
    plot_hloci_bands(
        selected[band_lo_col].to_numpy(dtype=float),
        selected[band_hi_col].to_numpy(dtype=float),
        directions,
        peak=selected[peak_col].to_numpy(dtype=float),
        mass=selected[mass_col].to_numpy(dtype=float),
        labels=selected[label_col],
        sort=None,
        up_color=up_color,
        down_color=down_color,
        alpha_range=alpha_range,
        mass_colorbar=False,
        ax=band_axis,
        xlabel="",
        ylabel=(ylabel if ylabel is not None
                else f"top {n_rows} {analyte_label} by FDR (sorted by band centre)"),
        title="\n".join([title_parts[0], " · ".join(title_parts[1:])]),
        title_fontsize=11,
    )
    band_axis.tick_params(axis="y", labelsize=6)
    if layer_limits is not None:
        band_axis.set_xlim(*layer_limits)
    if layer_to_distance is not None:
        add_border_distance_axis(band_axis, layer_to_distance)

    y = np.arange(n_rows)
    floor = float(fdr_floor) if fdr_floor is not None else 1e-300
    fdr = np.clip(selected[fdr_col].to_numpy(dtype=float), floor, 1.0)
    fdr_axis.barh(y, -np.log10(fdr), height=0.6, color="0.55", alpha=0.85, zorder=3)
    fdr_axis.axvline(-np.log10(fdr_threshold), ls="--", lw=1.0, color="0.35", zorder=2,
                       label=f"FDR = {fdr_threshold:g}")
    fdr_axis.set_xlabel(r"$-\log_{10}$ FDR")
    fdr_axis.set_title("significance", fontsize=11)
    fdr_axis.grid(axis="x", ls=":", lw=0.6, color="0.85", zorder=0)
    fdr_axis.tick_params(labelleft=False)
    fdr_axis.legend(frameon=False, fontsize=7, loc="lower right")

    mass = selected[mass_col].to_numpy(dtype=float)
    mass_min, mass_max = float(mass.min()), float(mass.max())
    norm = Normalize(vmin=mass_min, vmax=mass_max if mass_max > mass_min else mass_min + 1.0)

    def _ramp(color, n=256):
        red, green, blue = to_rgb(color)
        alpha = np.linspace(alpha_range[0], alpha_range[1], n)
        return ListedColormap(np.column_stack([
            np.full(n, red), np.full(n, green), np.full(n, blue), alpha,
        ]))

    if direction_labels is not None:
        directions_for_colorbar = [(str(_n), _c) for _n, _c in direction_labels]
    else:
        peak = selected[peak_col].to_numpy(dtype=float)
        directions_for_colorbar = []
        if np.any(peak > 0):
            directions_for_colorbar.append(("stroma (L > 0)", up_color))
        if np.any(peak <= 0):
            directions_for_colorbar.append(("tumour (L ≤ 0)", down_color))
    for index, (name, color) in enumerate(directions_for_colorbar):
        x0 = (0.04, 0.58)[min(index, 1)]
        color_axis = colorbar_axis.inset_axes([x0, colorbar_y, 0.34, 0.14])
        mapper = ScalarMappable(norm=norm, cmap=_ramp(color))
        mapper.set_array([])
        colorbar = fig.colorbar(mapper, cax=color_axis, orientation="horizontal")
        colorbar.ax.xaxis.set_ticks_position("bottom")
        colorbar.set_label("cluster mass", fontsize=6, labelpad=-1)
        colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
        colorbar_axis.text(
            x0 + 0.17, colorbar_y + 0.20, name, transform=colorbar_axis.transAxes,
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )

    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")

    return {
        "figure": fig,
        "band_axis": band_axis,
        "fdr_axis": fdr_axis,
        "colorbar_axis": colorbar_axis,
        "selected": selected,
        "n_candidates": n_candidates,
    }


def _hloci_bidir_order(n, elev_center, depr_center, elev_mass, depr_mass, sort_by):
    """Row ordering for the bidirectional H-Loci Summary.

    Returns an index array. Missing centres sort last (``+inf``) so genes with a
    band are shown before genes without one, for every keyed mode. ``barh``
    draws row 0 at the bottom, so the returned order is applied as-is (row i of
    the order → y-position i) and the *first* entry ends at the bottom.
    """
    if not sort_by:
        return np.arange(n)
    ec = np.where(np.isfinite(elev_center), elev_center, np.inf)
    dc = np.where(np.isfinite(depr_center), depr_center, np.inf)
    key = str(sort_by).replace("-", "_")
    if key == "elevated_center":
        k = ec
    elif key == "depressed_center":
        k = dc
    elif key == "midpoint":
        both = np.isfinite(elev_center) & np.isfinite(depr_center)
        k = np.where(both, (np.nan_to_num(elev_center) + np.nan_to_num(depr_center)) / 2.0,
                     np.where(np.isfinite(elev_center), elev_center,
                              np.where(np.isfinite(depr_center), depr_center, np.inf)))
    elif key == "dominant_center":
        em = np.nan_to_num(elev_mass, nan=0.0) if elev_mass is not None else np.zeros(n)
        dm = np.nan_to_num(depr_mass, nan=0.0) if depr_mass is not None else np.zeros(n)
        k = np.where(em >= dm, ec, dc)
        # if the dominant direction has no centre, fall back to the other
        k = np.where(np.isfinite(k), k, np.where(np.isfinite(ec), ec, dc))
    else:
        raise ValueError(
            "sort_by must be one of 'dominant_center', 'elevated_center', "
            "'depressed_center', 'midpoint', or None."
        )
    return np.argsort(k, kind="stable")


def plot_hloci_bands_bidir(
    labels,
    elev_lo,
    elev_hi,
    depr_lo,
    depr_hi,
    *,
    elev_center=None,
    depr_center=None,
    elev_mass=None,
    depr_mass=None,
    sort_by="dominant_center",
    ax=None,
    up_color="#d62728",
    down_color="#1f77b4",
    bar_height=0.6,
    alpha=0.85,
    edge_color="none",
    edge_width=0.0,
    center_marker=True,
    center_marker_color="0.15",
    center_marker_width=1.4,
    boundary_line=True,
    boundary_line_color="0.5",
    boundary_line_width=0.9,
    grid_axis="x",
    xlabel="border layer L",
    ylabel=None,
    title=None,
    title_fontsize=None,
):
    """Draw a **bidirectional H-Loci Summary**: two bands per gene row.

    Each gene occupies a single row on which up to two directional H-domains are
    drawn: the **elevated** band (``up_color``) and the **depressed** band
    (``down_color``), each with its own centre-of-mass tick. Genes for which a
    direction is absent (non-finite bounds) simply omit that band; a gene with
    neither band draws an empty row. This is the visual counterpart of
    :func:`hplot.stats.gradient_cluster_mass_screen` in
    ``band_mode="bidirectional"``.

    A depressed band on one side and an elevated band on the other may be the
    two ends of a single monotonic gradient rather than two independent
    programmes — this panel shows *where* each directional domain sits, it does
    not assert independence.

    Parameters
    ----------
    labels : sequence, shape (n,)
        Gene / feature row labels (one per row).
    elev_lo, elev_hi : array-like, shape (n,)
        Elevated-band lower / upper layer bounds. Non-finite → no elevated band.
    depr_lo, depr_hi : array-like, shape (n,)
        Depressed-band lower / upper layer bounds. Non-finite → no depressed band.
    elev_center, depr_center : array-like | None
        Per-direction centre-of-mass layer (tick position). Defaults to the band
        midpoint when omitted.
    elev_mass, depr_mass : array-like | None
        Per-direction cluster mass; required for ``sort_by="dominant_center"``.
    sort_by : {"dominant_center", "elevated_center", "depressed_center", \
"midpoint"} | None
        Row ordering key (see :func:`_hloci_bidir_order`). ``None`` keeps the
        caller-supplied order.
    ax : matplotlib.axes.Axes | None
        Axis to draw into; created when ``None``.
    up_color, down_color : str
        Elevated / depressed band colours.
    bar_height, alpha, edge_color, edge_width : float, float, str, float
        Bar styling.
    center_marker, center_marker_color, center_marker_width : bool, str, float
        Centre-of-mass tick styling.
    boundary_line, boundary_line_color, boundary_line_width : bool, str, float
        Layer-0 boundary reference styling.
    grid_axis : str | None
        Light-grid axis (``"x"``/``"y"``/``"both"``/``None``).
    xlabel, ylabel, title, title_fontsize : str | None, str | None, str | None, float | None
        Labels / title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis drawn into.
    """
    elo = np.asarray(elev_lo, dtype=float)
    ehi = np.asarray(elev_hi, dtype=float)
    dlo = np.asarray(depr_lo, dtype=float)
    dhi = np.asarray(depr_hi, dtype=float)
    n = len(elo)
    for arr in (ehi, dlo, dhi):
        if len(arr) != n:
            raise ValueError("all band-bound arrays must have the same length.")
    labels = list(labels)
    if len(labels) != n:
        raise ValueError("labels must match the number of rows.")
    ec = np.asarray(elev_center, dtype=float) if elev_center is not None else (elo + ehi) / 2.0
    dc = np.asarray(depr_center, dtype=float) if depr_center is not None else (dlo + dhi) / 2.0
    em = np.asarray(elev_mass, dtype=float) if elev_mass is not None else None
    dm = np.asarray(depr_mass, dtype=float) if depr_mass is not None else None

    order = _hloci_bidir_order(n, ec, dc, em, dm, sort_by)
    elo, ehi, dlo, dhi = elo[order], ehi[order], dlo[order], dhi[order]
    ec, dc = ec[order], dc[order]
    labels = [labels[i] for i in order]

    if ax is None:
        fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 18.0))
        _, ax = plt.subplots(figsize=(5.8, fig_h))

    y = np.arange(n)
    for _yi, _elo, _ehi, _ec, _dlo, _dhi, _dc in zip(y, elo, ehi, ec, dlo, dhi, dc):
        if np.isfinite(_elo) and np.isfinite(_ehi):
            ax.barh(_yi, abs(_ehi - _elo), left=min(_elo, _ehi), height=bar_height,
                    color=up_color, alpha=alpha, edgecolor=edge_color,
                    linewidth=edge_width, zorder=3)
            if center_marker and np.isfinite(_ec):
                ax.plot([_ec, _ec], [_yi - bar_height / 2.0, _yi + bar_height / 2.0],
                        color=center_marker_color, lw=center_marker_width, zorder=4)
        if np.isfinite(_dlo) and np.isfinite(_dhi):
            ax.barh(_yi, abs(_dhi - _dlo), left=min(_dlo, _dhi), height=bar_height,
                    color=down_color, alpha=alpha, edgecolor=edge_color,
                    linewidth=edge_width, zorder=3)
            if center_marker and np.isfinite(_dc):
                ax.plot([_dc, _dc], [_yi - bar_height / 2.0, _yi + bar_height / 2.0],
                        color=center_marker_color, lw=center_marker_width, zorder=4)

    if boundary_line:
        ax.axvline(0.0, color=boundary_line_color, lw=boundary_line_width,
                   ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
    if grid_axis:
        ax.grid(axis=grid_axis, color="0.9", lw=0.6)
    return ax


def _hpathway_cluster_mass_peak(row, layers):
    """Peak layer inside a pathway row's strongest contiguous above-median run.

    Mirrors the cluster-mass logic used by the spatial screen: find the run of
    contiguous above-median layers with the greatest summed elevation, then
    return the layer of the maximum inside that run (not a bare whole-row
    argmax). Degenerate/flat rows fall back to the plain argmax.
    """
    r = np.asarray(row, dtype=float)
    layers = np.asarray(layers)
    if not np.isfinite(r).any():
        return -np.inf
    thr = np.nanmedian(r)
    above = np.where(np.isfinite(r), r - thr, 0.0)
    above = np.where(above > 0.0, above, 0.0)
    best_sum, best_lo, best_hi = -np.inf, None, None
    i, n = 0, len(r)
    while i < n:
        if above[i] > 0.0:
            j, s = i, 0.0
            while j < n and above[j] > 0.0:
                s += above[j]; j += 1
            if s > best_sum:
                best_sum, best_lo, best_hi = s, i, j - 1
            i = j
        else:
            i += 1
    if best_lo is None:
        return float(layers[int(np.nanargmax(np.where(np.isfinite(r), r, -np.inf)))])
    seg = np.where(np.isfinite(r[best_lo:best_hi + 1]), r[best_lo:best_hi + 1], -np.inf)
    return float(layers[best_lo + int(np.argmax(seg))])


def plot_hpathway_dotplot(
    grid_df,
    *,
    score_col="score",
    fdr_col="fdr_dev",
    path_col="pathway",
    layer_col="layer",
    fdr_threshold=0.05,
    fdr_label=None,
    select_fdr_below=None,
    max_rows=40,
    layer_limits=None,
    layer_to_distance=None,
    size_range=(12.0, 400.0),
    size_mode="row",
    size_vmin=None,
    size_vmax=None,
    side_colorbar=True,
    cell_in=0.30,
    alpha_range=(0.25, 1.0),
    neglog_fdr_cap=3.0,
    order_by_peak=True,
    order_by="peak",
    order_shrink=0.25,
    row_order=None,
    direction_col=None,
    direction_labels=None,
    elevated_color="#d62728",
    depressed_color="#1f77b4",
    nodir_color="#d62728",
    direction_alpha=0.9,
    ax=None,
    title=None,
    savepath=None,
    dpi=240,
):
    """H-Pathway Summary: signature activity across the signed border axis.

    A dotplot over a (pathway x layer) grid. When a signed ``direction_col`` is
    supplied, dot **colour** encodes the direction (``elevated_color`` for a
    positive value, ``depressed_color`` for a negative one, ``nodir_color`` for
    0/NaN); dot **size** encodes the row-relative score within each pathway; and
    a black **ring** marks cells with FDR below ``fdr_threshold``. When no
    direction column is given every dot uses a single neutral colour
    (``nodir_color``); dot **size** still encodes the row-relative score, dot
    **alpha** still encodes ``-log10(FDR)`` and the **ring** still marks
    FDR < ``fdr_threshold`` (there is no elevated/depressed axis to colour by).
    Pathways are ordered by the position of their cluster-mass peak.

    The input ``grid_df`` is a tidy table with one row per (pathway, layer),
    such as the H-Pathway Summary grid: it must contain ``path_col``, ``layer_col``,
    ``score_col`` and the selected ``fdr_col`` (e.g. ``fdr_dev``,
    ``fdr_contrast``, ``fdr_treatment`` or ``fdr_strata4``).

    Parameters
    ----------
    grid_df : pandas.DataFrame
        Tidy (pathway x layer) grid with score and FDR columns.
    select_fdr_below : float | None
        When set, keep only pathways with FDR below this value in >= 1 shown
        layer (discovery-mode selection). ``None`` keeps every pathway.
    max_rows : int | None
        Cap on the number of pathways drawn (best min-FDR first).
    cell_in : float
        Physical size (inches) of one grid cell, shared by both axes, so the
        x-tick (layer) and y-tick (pathway) gaps are equal (square cells) and
        consistent across panels regardless of how many rows/columns are shown.
    layer_to_distance : Mapping[int, float] | None
        Optional physical-distance map from :func:`build_layer_distance_map`;
        when supplied the x-axis gains µm tick labels via
        :func:`add_border_distance_axis`.
    direction_col : str | None
        Optional signed column (e.g. ``dir_contrast`` from
        :func:`hplot.hpathway_summary_grid`, = mean(group2) − mean(group1)).
        When given, the dot **fill colour** encodes direction: ``elevated_color``
        for a positive value, ``depressed_color`` for a negative one, and
        ``nodir_color`` for 0/NaN. Dots stay circular; size still encodes the
        row-relative score and the black ring still marks FDR < ``fdr_threshold``.
    direction_labels : tuple[str, str] | None
        ``(depressed_label, elevated_label)`` for the direction legend, e.g.
        ``("depressed vs baseline", "elevated vs baseline")``. Defaults to
        ``("depressed", "elevated")``.
    elevated_color, depressed_color : str
        Fill colours for positive (elevated) and negative (depressed) direction.
    nodir_color : str
        Fill colour for cells with no signed direction (0/NaN).
    direction_alpha : float
        Peak dot opacity in direction mode. Dot alpha ramps with ``-log10(FDR)``
        over ``alpha_range`` and is rescaled so the most-significant dot reaches
        this value; colour encodes direction and the ring marks FDR < threshold.
    savepath : str | pathlib.Path | None
        Optional PNG output path. An SVG sibling is also written.

    Returns
    -------
    dict | None
        ``{"figure", "ax", "colorbar_axis", "selected"}``; ``None`` when no
        pathway passes selection.
    """
    import matplotlib
    from pathlib import Path
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # The significance channel is not always an FDR: a permutation p or an uncorrected
    # per-cell p is legitimate here provided the legend says so rather than implying a
    # correction that was never applied.
    _sig_name = fdr_label if fdr_label else "FDR"

    piv_s = grid_df.pivot(index=path_col, columns=layer_col, values=score_col)
    _has_fdr = fdr_col is not None
    if _has_fdr:
        piv_f = grid_df.pivot(index=path_col, columns=layer_col, values=fdr_col)
    else:
        piv_f = piv_s.copy()
        piv_f[:] = np.nan
    layers = np.array(sorted(piv_s.columns))
    if layer_limits is not None:
        layers = layers[(layers >= layer_limits[0]) & (layers <= layer_limits[1])]
    piv_s = piv_s[layers]
    piv_f = piv_f.reindex(columns=layers)

    # ---- auto-selection: keep pathways significant in >= 1 shown layer ----
    if select_fdr_below is not None:
        if not _has_fdr:
            raise ValueError("select_fdr_below needs an fdr_col; got fdr_col=None.")
        keep = [p for p in piv_f.index
                if np.isfinite(piv_f.loc[p].to_numpy(dtype=float)).any()
                and np.nanmin(piv_f.loc[p].to_numpy(dtype=float)) < select_fdr_below]
        piv_s = piv_s.loc[keep]
        piv_f = piv_f.loc[keep]
    # cap rows so a big discovery stays readable: best min-FDR first when there is
    # an FDR channel, else the strongest rows by peak score
    if max_rows is not None and len(piv_s.index) > max_rows:
        if _has_fdr:
            best = {p: np.nanmin(piv_f.loc[p].to_numpy(dtype=float)) for p in piv_f.index}
            order = sorted(piv_f.index, key=lambda p: best[p])[:max_rows]
        else:
            peak = {p: np.nanmax(piv_s.loc[p].to_numpy(dtype=float)) for p in piv_s.index}
            order = sorted(piv_s.index, key=lambda p: -peak[p])[:max_rows]
        piv_s = piv_s.loc[order]
        piv_f = piv_f.loc[order]

    paths = list(piv_s.index)
    if len(paths) == 0:
        print(f"H-Pathway Summary [{fdr_col}]: no pathway passed selection "
              f"(FDR < {select_fdr_below}); nothing to plot.")
        return None
    if order_by_peak:
        # Ordering keys off the column actually drawn, so the sort the reader infers from
        # dot size is the sort that was applied. "centroid" is the mass-weighted mean
        # layer: continuous, so rows spread smoothly instead of piling onto the handful
        # of layers that happen to win an argmax.
        if str(order_by) == "centroid":
            _w_all, _cen_all = {}, {}
            for p in paths:
                v = piv_s.loc[p].to_numpy(dtype=float)
                w = np.where(np.isfinite(v), v, np.nan)
                w = w - np.nanmin(w)
                w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                _w_all[p] = float(np.sum(w))
                _cen_all[p] = (float(np.sum(w * layers) / np.sum(w))
                               if np.sum(w) > 0 else float(np.mean(layers)))
            # A row with almost no drawn signal still has a centroid, but it is set by
            # noise -- and would otherwise let an empty row claim the top of the panel.
            # Shrink toward the panel's median centroid with a weight of `order_shrink`
            # x the median row mass: rows carrying real signal barely move, empty ones
            # collapse to the middle of the ordering where they make no spatial claim.
            # The neutral point is the median centroid rather than the axis midpoint,
            # which on an asymmetric window (e.g. -5..+15) is itself an extreme position.
            _mid = float(np.median(list(_cen_all.values()) or [0.0]))
            _k = float(order_shrink) * float(np.median(list(_w_all.values()) or [0.0]))

            def _key(p):
                m = _w_all[p]
                return (m * _cen_all[p] + _k * _mid) / (m + _k) if (m + _k) > 0 else _mid
        elif str(order_by) == "peak":
            def _key(p):
                return _hpathway_cluster_mass_peak(
                    piv_s.loc[p].to_numpy(dtype=float), layers)
        else:
            raise ValueError(f"order_by must be 'peak' or 'centroid'; got {order_by!r}.")
        # y increases upward, so an ascending sort renders bottom -> top as inner -> outer.
        paths = sorted(paths, key=_key)
    if row_order is not None:
        # An explicit order lets the caller rank rows on a quantity the panel does not
        # carry -- e.g. a peak read off the chance-standardised profile rather than the
        # raw one that is drawn.
        _want = [p for p in row_order if p in set(paths)]
        paths = _want + [p for p in paths if p not in set(_want)]
    piv_s = piv_s.reindex(index=paths)
    piv_f = piv_f.reindex(index=paths)

    S = piv_s.to_numpy(dtype=float)
    F = piv_f.to_numpy(dtype=float)

    # optional signed direction (e.g. dir_contrast = mean(group2) - mean(group1))
    D = None
    if direction_col is not None and direction_col in grid_df.columns:
        piv_d = grid_df.pivot(index=path_col, columns=layer_col, values=direction_col)
        D = piv_d.reindex(index=paths, columns=layers).to_numpy(dtype=float)

    # Row-relative sizing: normalize each pathway row to [0, 1] independently so
    # within-pathway layer differences span the full size range.
    if str(size_mode) == "absolute":
        # Dot area means the same thing in every row -- required when the score is
        # already expressed in units of chance (e.g. ratio_vs_null), where a value of
        # 3 in one row and 1.2 in another must not render identically.
        _lo = float(np.nanmin(S)) if size_vmin is None else float(size_vmin)
        _hi = float(np.nanmax(S)) if size_vmax is None else float(size_vmax)
        if not np.isfinite(_lo) or not np.isfinite(_hi) or _hi <= _lo:
            _hi = _lo + 1.0
        S_relative = np.clip((S - _lo) / (_hi - _lo), 0.0, 1.0)
    elif str(size_mode) == "row":
        row_lo = np.nanmin(S, axis=1, keepdims=True)
        row_hi = np.nanmax(S, axis=1, keepdims=True)
        row_span = row_hi - row_lo
        S_relative = np.divide(S - row_lo, row_span, out=np.full_like(S, 0.5),
                               where=row_span > 1e-9)
    else:
        raise ValueError(f"size_mode must be 'row' or 'absolute'; got {size_mode!r}.")

    xs, ys, sizes, alphas, rings = [], [], [], [], []
    dirs = []
    for iy, p in enumerate(paths):
        for ix, L in enumerate(layers):
            if not np.isfinite(S[iy, ix]):
                continue
            xs.append(L)
            ys.append(iy)
            sizes.append(np.interp(S_relative[iy, ix], (0.0, 1.0), size_range))
            fv = F[iy, ix]
            if not _has_fdr:
                a = alpha_range[1]
            elif np.isfinite(fv) and fv > 0:
                a = np.interp(-np.log10(fv), (0.0, neglog_fdr_cap), alpha_range)
            else:
                a = alpha_range[0]
            alphas.append(float(np.clip(a, alpha_range[0], alpha_range[1])))
            rings.append(bool(np.isfinite(fv) and fv < fdr_threshold))
            dirs.append(float(D[iy, ix]) if D is not None else np.nan)

    ax_cbar = None
    fig = None
    if ax is None:
        # ---- deterministic square-cell layout --------------------------------
        # One knob (`cell_in`) sets BOTH the x-tick (layer) and y-tick (pathway)
        # spacing, so cells are square and the visual gap is identical across
        # panels regardless of how many pathways/layers each selects. The data
        # axes are sized in inches and fixed inch margins are added for the tick
        # labels, legend, title and colour-key, decoupling the grid from its
        # decorations.
        _ax_w = cell_in * max(len(layers), 1)     # data-area width  (inches)
        _ax_h = cell_in * max(len(paths), 1)      # data-area height (inches)

        # The two alpha-ramp column titles are caller-supplied and can be long
        # ("elevated vs own-window baseline"). Fit them to their column here,
        # before the figure height is fixed, so they can never collide.
        if direction_labels and len(direction_labels) == 2:
            _dn_lbl, _up_lbl = direction_labels
        else:
            _dn_lbl, _up_lbl = "depressed", "elevated"
        _fs_title = 10.5
        _col_in = (_KEY_CB_W + _KEY_GAP) * _ax_w
        while _fs_title > 6.5:
            _budget = max(6, int(_col_in / (_fs_title * 0.60 / 72.0)))
            if max(len(_up_lbl), len(_dn_lbl)) <= _budget:
                break
            _fs_title -= 0.5
        _budget = max(6, int(_col_in / (_fs_title * 0.60 / 72.0)))
        _up_wrap = textwrap.wrap(_up_lbl, _budget) or [_up_lbl]
        _dn_wrap = textwrap.wrap(_dn_lbl, _budget) or [_dn_lbl]
        _key_rows = max(len(_up_wrap), len(_dn_wrap))
        _up_short, _dn_short = "\n".join(_up_wrap), "\n".join(_dn_wrap)

        # the bottom key row replaces the right-hand legends, so the right
        # margin only has to clear the last x tick label
        _ml, _mt = 2.8, 0.9                       # left ylabels | top title
        _mr = 0.6 if side_colorbar else 2.4
        if side_colorbar:
            _key_in = 0.95 + 0.17 * (_key_rows - 1)   # colour-key strip height (inches)
            _gap_in = 0.66                        # border-distance axis + its title
            _mb = _gap_in + _key_in               # bottom margin (inches)
            fig_w, fig_h = _ml + _ax_w + _mr, _mb + _ax_h + _mt
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_axes([_ml / fig_w, _mb / fig_h, _ax_w / fig_w, _ax_h / fig_h])
            ax_cbar = fig.add_axes([_ml / fig_w, 0.0, _ax_w / fig_w, _key_in / fig_h])
            ax_cbar.set_axis_off()
        else:
            _mb = 0.9
            fig_w, fig_h = _ml + _ax_w + _mr, _mb + _ax_h + _mt
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_axes([_ml / fig_w, _mb / fig_h, _ax_w / fig_w, _ax_h / fig_h])
    else:
        fig = ax.figure

    _xs = np.asarray(xs, dtype=float)
    _ys = np.asarray(ys, dtype=float)
    _sz = np.asarray(sizes, dtype=float)
    _rg = np.asarray(rings, dtype=bool)
    _dr = np.asarray(dirs, dtype=float)

    # Fill colour. With a signed direction the colour encodes it: elevated (one
    # colour) vs depressed (the other); cells with no signed direction (0/NaN)
    # are neutral grey. Dots are solid circles — size encodes the row-relative
    # score and a black RING marks FDR < threshold, so colour is the sole
    # direction channel (no triangles). Without a direction column (e.g. a
    # >2-group contrast that has no elevated/depressed axis) every dot is a
    # single neutral colour; size + alpha (FDR) + ring still apply, so the
    # border-side (tumour/stroma) colouring is not used.
    rgba = np.zeros((len(xs), 4))
    if D is not None:
        _el_rgb = matplotlib.colors.to_rgb(elevated_color)
        _de_rgb = matplotlib.colors.to_rgb(depressed_color)
        # in direction mode a cell with no signed direction is ambiguous, not
        # "directionless", so it stays grey whatever `nodir_color` is set to
        _nd_rgb = matplotlib.colors.to_rgb("0.7")
        for _k in range(len(xs)):
            _d = _dr[_k]
            if np.isfinite(_d) and _d > 0:
                rgba[_k, :3] = _el_rgb
            elif np.isfinite(_d) and _d < 0:
                rgba[_k, :3] = _de_rgb
            else:
                rgba[_k, :3] = _nd_rgb
        # alpha encodes FDR (opacity ramps with -log10(FDR), same scale as the
        # legacy mode), rescaled so the most-significant dot reaches
        # `direction_alpha`; colour still encodes direction and the ring still
        # marks FDR < threshold.
        _amax = max(float(alpha_range[1]), 1e-6)
        rgba[:, 3] = np.asarray(alphas, dtype=float) * (float(direction_alpha) / _amax)
    else:
        # no signed direction available -> neutral fill for every dot; alpha
        # still encodes FDR and the ring marks significance (no tumour/stroma
        # side colouring).
        _nd_rgb = matplotlib.colors.to_rgb(nodir_color)
        for _k in range(len(xs)):
            rgba[_k, :3] = _nd_rgb
        rgba[:, 3] = alphas

    ax.scatter(_xs, _ys, s=_sz, c=rgba, linewidths=0, zorder=3)
    # black ring = FDR < threshold: a same-size circular outline over every
    # significant dot.
    if _rg.any():
        ax.scatter(_xs[_rg], _ys[_rg], s=_sz[_rg], marker="o",
                   facecolors="none", edgecolors="k", linewidths=1.5, zorder=4)

    ax.axvline(0.0, color="0.4", ls="--", lw=1.8, zorder=1)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels(paths, fontsize=9)
    ax.set_ylim(-0.6, len(paths) - 0.4)
    if layer_limits is not None:
        ax.set_xlim(layer_limits[0] - 0.6, layer_limits[1] + 0.6)
    ax.set_xlabel("border layer L  (≤ 0 tumour | > 0 stroma)", fontsize=12)
    ax.grid(axis="both", color="0.92", lw=0.5, zorder=0)
    ax.tick_params(axis="both", pad=2, labelsize=10)
    if title:
        ax.set_title(title, fontsize=13)
    if layer_to_distance is not None:
        add_border_distance_axis(ax, layer_to_distance)

    # size legend (neutral grey; encodes the row-relative dot-size scale only).
    # Skipped when the bottom colour-key row is drawn, which carries the same
    # information -- otherwise every figure gets two legends.
    if ax_cbar is None:
        size_ref = [0.0, 0.5, 1.0]
        size_labels = ["row min", "row mid", "row max"]
        size_handles = [Line2D([0], [0], marker="o", linestyle="none",
                               markerfacecolor="0.5", markeredgecolor="none",
                               markersize=np.sqrt(np.interp(v, (0.0, 1.0), size_range)),
                               label=lbl) for v, lbl in zip(size_ref, size_labels)]
        leg1 = ax.legend(handles=size_handles, title="relative score", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), frameon=False, labelspacing=1.1,
                         fontsize=10, title_fontsize=11)
        ax.add_artist(leg1)

    # direction key (only when a signed direction is drawn): colour = elevated /
    # depressed / no direction, plus a ringed dot for the FDR ring.
    if D is not None and ax_cbar is None:
        if direction_labels and len(direction_labels) == 2:
            _dn_lbl, _up_lbl = direction_labels
        else:
            _dn_lbl, _up_lbl = "depressed", "elevated"
        _dir_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=elevated_color,
                   markeredgecolor="none", markersize=9, label=_up_lbl),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=depressed_color,
                   markeredgecolor="none", markersize=9, label=_dn_lbl),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=nodir_color,
                   markeredgecolor="none", markersize=9, label="no direction"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.85",
                   markeredgecolor="k", markeredgewidth=0.9, markersize=9,
                   label=f"FDR < {fdr_threshold:g}"),
        ]
        leg_dir = ax.legend(handles=_dir_handles, title="Direction / significance",
                            loc="center left", bbox_to_anchor=(1.01, 0.55), frameon=False,
                            fontsize=10, title_fontsize=11)
        ax.add_artist(leg_dir)

        # opacity key: neutral-grey dots at graduated opacity mapped to FDR, so
        # the alpha channel reads like a normal dotplot legend (fainter = less
        # significant). Uses the same alpha mapping as the drawn dots.
        _a_scale = float(direction_alpha) / max(float(alpha_range[1]), 1e-6)
        _fdr_ref = [fdr_threshold, 0.01, 1.0 / (10 ** neglog_fdr_cap)]
        _alpha_handles = []
        for _q in _fdr_ref:
            _a = float(np.clip(np.interp(-np.log10(_q), (0.0, neglog_fdr_cap), alpha_range),
                               alpha_range[0], alpha_range[1])) * _a_scale
            _alpha_handles.append(Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=(0.35, 0.35, 0.35, float(np.clip(_a, 0.0, 1.0))),
                markeredgecolor="none", markersize=9, label=f"FDR = {_q:g}"))
        leg_alpha = ax.legend(handles=_alpha_handles, title="Opacity = FDR",
                              loc="lower left", bbox_to_anchor=(1.01, 0.05),
                              frameon=False, fontsize=10, title_fontsize=11)
        ax.add_artist(leg_alpha)

    # FDR key row below the panel. Direction mode shows:
    # (1) dot-size gradient, (2) two side-by-side alpha-ramp colorbars (red for
    # elevated, blue for depressed) — each showing the actual color with FDR
    # opacity ramp, (3) ring note. No-direction mode shows a single neutral
    # opacity ramp. When no colour-key row is available (external ax /
    # side_colorbar=False) a discrete right-side legend is the fallback.
    if ax_cbar is not None and not _has_fdr:
        # no significance channel at all: size is the only encoded quantity, so the
        # strip must not imply an FDR ramp or a ring that is never drawn.
        _sz_x0, _sz_step = 0.34, 0.09
        if str(size_mode) == "absolute":
            _lo = float(np.nanmin(S)) if size_vmin is None else float(size_vmin)
            _hi = float(np.nanmax(S)) if size_vmax is None else float(size_vmax)
            _vals = [0.0, 0.5, 1.0]
            _labs = [f"{_lo + v * (_hi - _lo):.3g}" for v in _vals]
            _cap = "size = score (absolute)  \u00b7  no significance channel"
        else:
            _vals, _labs = [0.0, 0.5, 1.0], ["0", "0.5", "1"]
            _cap = "size = row-relative score  \u00b7  no significance channel"
        for _k, (_v, _lbl) in enumerate(zip(_vals, _labs)):
            _ms = np.sqrt(np.interp(_v, (0.0, 1.0), size_range))
            _x = _sz_x0 + _sz_step * _k
            ax_cbar.plot(_x, 0.64, "o", color=nodir_color, markersize=_ms,
                         transform=ax_cbar.transAxes, clip_on=False)
            ax_cbar.text(_x, 0.24, _lbl, transform=ax_cbar.transAxes, ha="center",
                         va="bottom", fontsize=9.0, color="0.35", clip_on=False)
        ax_cbar.text(0.5, 0.86, _cap,
                     transform=ax_cbar.transAxes, ha="center", va="bottom",
                     fontsize=10.5, color="0.35")
        # A group contrast still has a legitimate side (which arm is higher), so the
        # colours are named even though there is no significance channel.
        if D is not None:
            if direction_labels and len(direction_labels) == 2:
                _dn_lbl, _up_lbl = direction_labels
            else:
                _dn_lbl, _up_lbl = "depressed", "elevated"
            for _k, (_c, _lbl) in enumerate(((elevated_color, _up_lbl),
                                             (depressed_color, _dn_lbl))):
                _x = 0.62 + 0.19 * _k
                ax_cbar.plot(_x, 0.64, "o", color=_c, markersize=np.sqrt(size_range[1]),
                             transform=ax_cbar.transAxes, clip_on=False)
                ax_cbar.text(_x, 0.24, _lbl, transform=ax_cbar.transAxes, ha="center",
                             va="bottom", fontsize=8.5, color=_c, clip_on=False)
    elif ax_cbar is not None:
        # direction mode: full legend strip with size gradient and two colored
        # alpha-ramp colorbars (elevated = red, depressed = blue)
        _norm = Normalize(vmin=0.0, vmax=neglog_fdr_cap)
        _yy = 0.58          # bar bottom (fraction of ax_cbar)
        _sh = 0.20          # bar height (fraction of ax_cbar)
        _lab_y = 0.86       # shared y for all column titles (closer to top)
        _bot_y = 0.24       # shared y for all bottom captions (closer to elements above)
        # `_fs_title`, `_up_short` and `_dn_short` were fitted to the column
        # width during layout so the two ramp titles cannot overlap. Titles are
        # bottom-anchored, so a wrapped title has to start lower to stay inside
        # the (correspondingly taller) key strip.
        _lab_y = min(_lab_y, 0.99 - _key_rows * (_fs_title * 1.25 / 72.0) / _key_in)
        _fs_num = 9.0
        _fs_tick = 9.0

        # (1) dot-size legend. Spacing is set by whichever is wider, the largest
        # marker or its label, so the numbers cannot run together when size_range
        # or the panel width changes.
        _sz_x0 = 0.02
        _sz_vals = [0.0, 0.5, 1.0]
        if str(size_mode) == "absolute":
            _s_lo = float(np.nanmin(S)) if size_vmin is None else float(size_vmin)
            _s_hi = float(np.nanmax(S)) if size_vmax is None else float(size_vmax)
            _sz_labels = [f"{_s_lo + v * (_s_hi - _s_lo):.3g}" for v in _sz_vals]
            _sz_title = "score"
        else:
            _sz_labels = ["0", "0.5", "1"]
            _sz_title = "relative score"
        _key_w_in = ax_cbar.get_position().width * fig.get_figwidth()
        _step = (max(np.sqrt(size_range[1]),
                     _fs_num * 0.60 * max(len(_l) for _l in _sz_labels))
                 * 1.3 / 72.0) / max(_key_w_in, 1e-6)
        # Fit the key columns (sizes, one ramp per direction PRESENT, ring) inside the
        # strip. The size spacing gives way first, then the ramp width, so the columns
        # cannot collide however size_range and the panel width are set.
        _ramps = [(_c, _lab) for _present, _c, _lab in
                  ((D is not None and bool(np.any(D > 0)), elevated_color, _up_short),
                   (D is not None and bool(np.any(D < 0)), depressed_color, _dn_short))
                  if _present]
        if not _ramps:                      # no direction to encode: one neutral ramp,
            _ramps = [(nodir_color, "")]    # named by its own caption below
        _n_ramp = len(_ramps)
        _cb_w, _gap, _ring_w = _KEY_CB_W, _KEY_GAP, 0.10
        # The strip is laid out left-aligned, so any width left over would otherwise sit
        # as dead space after the ring column; spend it on the gaps instead.
        _avail = 0.98 - _sz_x0
        _need = 2 * _step + _n_ramp * _cb_w + (_n_ramp + 1) * _gap + _ring_w
        _over = _need - _avail
        if _over > 0:
            _step = max(0.030, _step - _over / 2.0)
            _over = (2 * _step + _n_ramp * _cb_w + (_n_ramp + 1) * _gap + _ring_w) - _avail
        if _over > 0:
            _cb_w = max(0.10, _cb_w - _over / max(_n_ramp, 1))
        else:
            _gap += (-_over) / (_n_ramp + 2)
        for _k, (_v, _lbl) in enumerate(zip(_sz_vals, _sz_labels)):
            _ms = np.sqrt(np.interp(_v, (0.0, 1.0), size_range))  # marker size in points
            _x = _sz_x0 + _step * _k
            ax_cbar.plot(_x, 0.64, "o", color="0.5", markersize=_ms, transform=ax_cbar.transAxes,
                         clip_on=False)
            ax_cbar.text(_x, _bot_y, _lbl, transform=ax_cbar.transAxes, ha="center", va="bottom",
                         fontsize=_fs_num, color="0.35", clip_on=False)
        ax_cbar.text(_sz_x0 + _step, _lab_y, _sz_title, transform=ax_cbar.transAxes,
                     ha="center", va="bottom", fontsize=_fs_title, color="0.25", clip_on=False)

        # (2) one alpha-ramp colorbar per direction actually drawn, so a single-direction
        # panel does not carry a ramp for a colour that never appears on it.
        _cb_x0 = _sz_x0 + 2 * _step + _gap
        for _ri, (_rc, _rlab) in enumerate(_ramps):
            _x0r = _cb_x0 + _ri * (_cb_w + _gap)
            _caxr = ax_cbar.inset_axes([_x0r, _yy, _cb_w, _sh])
            _smr = ScalarMappable(norm=_norm, cmap=_alpha_ramp_cmap(
                _rc, alpha_range[0], alpha_range[1]))
            _smr.set_array([])
            _cbr = fig.colorbar(_smr, cax=_caxr, orientation="horizontal")
            _cbr.ax.xaxis.set_ticks_position("bottom")
            _cbr.ax.tick_params(labelsize=_fs_tick, length=2, pad=1)
            _cbr.set_ticks([0, 1, 2, 3])
            ax_cbar.text(_x0r + _cb_w / 2, _lab_y, _rlab, transform=ax_cbar.transAxes,
                         ha="center", va="bottom", fontsize=_fs_title, color="0.25",
                         clip_on=False)
        _cb_end = _cb_x0 + _n_ramp * _cb_w + (_n_ramp - 1) * _gap

        # shared -log10 FDR label below the ramps (aligned with the size numbers)
        ax_cbar.text((_cb_x0 + _cb_end) / 2, _bot_y,
                     rf"$-\log_{{10}}$ {_sig_name}  (opacity)", transform=ax_cbar.transAxes,
                     ha="center", va="bottom", fontsize=_fs_num, color="0.35", clip_on=False)

        # (3) ring = FDR < threshold: an OPEN ringed dot (no fill), matching the size legend
        _ring_x = _cb_end + _gap + _ring_w / 2
        ax_cbar.text(_ring_x, _lab_y, "significance", transform=ax_cbar.transAxes,
                     ha="center", va="bottom", fontsize=_fs_title, color="0.25", clip_on=False)
        # markersize matches the "1" dot in the size legend
        _ring_ms = np.sqrt(size_range[1])
        ax_cbar.plot(_ring_x, 0.64, "o", markerfacecolor="none", markeredgecolor="k",
                     markeredgewidth=1.2, markersize=_ring_ms, transform=ax_cbar.transAxes,
                     clip_on=False)
        ax_cbar.text(_ring_x, _bot_y, f"{_sig_name} < {fdr_threshold:g}",
                     transform=ax_cbar.transAxes,
                     ha="center", va="bottom", fontsize=_fs_num, color="0.3", clip_on=False)
    elif ax_cbar is not None:
        # no direction: a single neutral opacity ramp (alpha = -log10 FDR); no
        # tumour/stroma side colouring.
        _norm = Normalize(vmin=0.0, vmax=neglog_fdr_cap)
        _sh, _cb_w = 0.20, 0.40
        _x0 = 0.30
        _yy = 0.58          # bar bottom (match direction mode)
        _cap_y = 0.86       # caption y (match direction mode _lab_y)
        _fs_title = 10.5    # match direction mode
        _fs_tick = 9.0      # match direction mode
        _cax = ax_cbar.inset_axes([_x0, _yy, _cb_w, _sh])
        _sm = ScalarMappable(norm=_norm, cmap=_alpha_ramp_cmap(
            nodir_color, alpha_range[0], alpha_range[1]))
        _sm.set_array([])
        _cb = fig.colorbar(_sm, cax=_cax, orientation="horizontal")
        _cb.ax.xaxis.set_ticks_position("bottom")
        _cb.set_label(r"$-\log_{10}$ FDR", fontsize=_fs_tick, labelpad=-1)
        _cb.ax.tick_params(labelsize=_fs_tick, length=2, pad=1)
        ax_cbar.text(0.5, _cap_y,
                     f"opacity = FDR  \u00b7  size = row-relative score"
                     f"  \u00b7  ring = FDR < {fdr_threshold:g}",
                     transform=ax_cbar.transAxes, ha="center", va="bottom",
                     fontsize=_fs_title, color="0.35")
    elif D is None:
        alpha_ref = [1.0, 0.05, 0.005]
        alpha_handles = [Line2D([0], [0], marker="o", linestyle="none",
                                markerfacecolor=(*matplotlib.colors.to_rgb(nodir_color), float(np.clip(
                                    np.interp(-np.log10(q), (0.0, neglog_fdr_cap), alpha_range),
                                    alpha_range[0], alpha_range[1]))),
                                markeredgecolor="k" if q < fdr_threshold else "none",
                                markeredgewidth=0.7, markersize=9,
                                label=f"FDR = {q:g}") for q in alpha_ref]
        ax.legend(handles=alpha_handles, title=f"Opacity = FDR ({fdr_col})", loc="lower left",
                  bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=10, title_fontsize=11)

    if savepath is not None:
        output = Path(savepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")

    return {"figure": fig, "ax": ax, "colorbar_axis": ax_cbar, "selected": paths}



def plot_hloci_dotplot(
    grid_df,
    *,
    feature_col="feature",
    layer_col="layer",
    score_col="score",
    fdr_col="fdr",
    direction_col="direction",
    fdr_threshold=0.05,
    select_fdr_below=None,
    max_rows=50,
    layer_limits=None,
    layer_to_distance=None,
    size_range=(12.0, 400.0),
    side_colorbar=True,
    cell_in=0.30,
    alpha_range=(0.25, 1.0),
    neglog_fdr_cap=3.0,
    order_by_peak=True,
    direction_labels=None,
    elevated_color="#d62728",
    depressed_color="#1f77b4",
    nodir_color="0.7",
    direction_alpha=0.9,
    ax=None,
    title=None,
    savepath=None,
    dpi=240,
):
    """H-Loci Summary (dotplot mode): feature activity across the signed border axis.

    A dotplot over a (feature x layer) grid, where features can be genes, cell
    types, or ligand-receptor pairs. This complements :func:`plot_hloci_bands`
    by showing the *full layer profile* rather than just the dominant cluster
    band — revealing secondary peaks that the band-mode visualization collapses.

    Visual encoding:
    - **Dot size** encodes the row-relative score (normalized within each feature)
    - **Dot colour** encodes direction when ``direction_col`` is supplied
      (``elevated_color`` for positive, ``depressed_color`` for negative)
    - **Dot alpha** ramps with ``-log10(FDR)``
    - **Black ring** marks cells with FDR below ``fdr_threshold``

    The input ``grid_df`` is a tidy table with one row per (feature, layer),
    similar to the H-Pathway Summary grid but for genes, cell types, or pairs.

    Parameters
    ----------
    grid_df : pandas.DataFrame
        Tidy (feature x layer) grid with score and FDR columns.
    feature_col : str
        Column name for the feature label (gene, cell type, or pair).
    layer_col : str
        Column name for the border layer index.
    score_col : str
        Column name for the activity/mass score at each layer.
    fdr_col : str
        Column name for the FDR/p-value at each layer.
    direction_col : str | None
        Optional signed column for direction (elevated vs depressed).
    select_fdr_below : float | None
        Keep only features with FDR below this value in >= 1 shown layer.
    max_rows : int | None
        Cap on the number of features drawn (best min-FDR first).
    layer_limits : tuple[float, float] | None
        (min_layer, max_layer) to display.
    layer_to_distance : Mapping[int, float] | None
        Physical-distance map for µm tick labels.
    direction_labels : tuple[str, str] | None
        ``(depressed_label, elevated_label)`` for the direction legend.
    savepath : str | pathlib.Path | None
        Optional PNG output path. An SVG sibling is also written.

    Returns
    -------
    dict | None
        ``{"figure", "ax", "colorbar_axis", "selected"}``; ``None`` when no
        feature passes selection.

    See Also
    --------
    plot_hloci_bands : Band-mode H-Loci Summary (shows dominant cluster only).
    plot_hpathway_dotplot : H-Pathway Summary dotplot for pathway signatures.
    """
    # Delegate to plot_hpathway_dotplot with H-Loci-appropriate column names.
    return plot_hpathway_dotplot(
        grid_df,
        path_col=feature_col,
        layer_col=layer_col,
        score_col=score_col,
        fdr_col=fdr_col,
        fdr_threshold=fdr_threshold,
        select_fdr_below=select_fdr_below,
        max_rows=max_rows,
        layer_limits=layer_limits,
        layer_to_distance=layer_to_distance,
        size_range=size_range,
        side_colorbar=side_colorbar,
        cell_in=cell_in,
        alpha_range=alpha_range,
        neglog_fdr_cap=neglog_fdr_cap,
        order_by_peak=order_by_peak,
        direction_col=direction_col,
        direction_labels=direction_labels,
        elevated_color=elevated_color,
        depressed_color=depressed_color,
        nodir_color=nodir_color,
        direction_alpha=direction_alpha,
        ax=ax,
        title=title,
        savepath=savepath,
        dpi=dpi,
    )
