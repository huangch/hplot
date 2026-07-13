import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

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
    band, i.e. the top panel that pairs with :func:`plot_delta_hplot_gam`. This
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


def plot_delta_hplot_gam(
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
        :math:`\Delta = 0` line, which is always drawn. Default ``True``.
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


def plot_hloci_summary(
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


def plot_hloci_bands(
    band_lo,
    band_hi,
    directions,
    *,
    peak=None,
    labels=None,
    sort="outer_to_inner",
    ax=None,
    up_color="#d62728",
    down_color="#1f77b4",
    default_color="0.6",
    bar_height=0.6,
    alpha=0.85,
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
):
    """Draw an **H-Loci Summary (band mode)** panel from cluster-band extents.

    Alternative rendering of the H-Loci Summary in which each row is one
    feature's **H-Locus** drawn as a *horizontal bar* spanning its actual
    cluster band ``[band_lo, band_hi]`` on the signed border-layer axis, filled
    by direction (``up_color`` for elevated, ``down_color`` for depressed). This
    is the bar-chart counterpart of the direction-shaded ``axvspan`` band used
    on the per-feature H-plots (i.e. it shows *where* the band sits and *how
    wide* it is — the cluster width — rather than mapping the mass to a strip
    thickness as :func:`plot_hloci_summary` does).

    Parameters
    ----------
    band_lo, band_hi : array-like, shape (n,)
        Lower / upper layer bounds of each feature's scored cluster band. Rows
        with a non-finite bound are skipped.
    directions : array-like, shape (n,)
        Per-feature direction, coerced by :func:`_hloci_up_mask` (accepts
        ``"elevated"``/``"depressed"``, bool, or signed numeric). ``up`` rows
        are filled with ``up_color``, ``down`` rows with ``down_color``.
    peak : array-like, shape (n,) | None
        Optional per-feature peak layer; when given a short vertical tick is
        drawn inside each bar at that layer.
    labels : sequence | None
        Row (y-tick) labels.
    sort : {"outer_to_inner", "inner_to_outer"} | None
        Default row ordering, keyed on the **peak centre of cluster mass**
        (``peak`` when supplied, else the band midpoint). ``"outer_to_inner"``
        (default) places the outermost H-Loci (stroma side, high layer L) at
        the top and the innermost (tumour interior, low L) at the bottom;
        ``"inner_to_outer"`` reverses this. Pass ``None`` (or ``False``) to
        keep the caller-supplied row order.
    ax : matplotlib.axes.Axes | None
        Axis to draw into; a new one is created when ``None``.
    up_color, down_color : str
        Fill colours for elevated / depressed bands.
    default_color : str
        Fallback fill when a direction cannot be resolved.
    bar_height : float
        Height of each horizontal bar (in data units; rows are unit-spaced).
    alpha : float
        Bar fill opacity.
    edge_color : str
        Bar edge colour. Default ``"none"``.
    edge_width : float
        Bar edge width.
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
    is_up = _hloci_up_mask(directions)
    if len(is_up) != n:
        raise ValueError("directions must match the length of band_lo.")
    pk = np.asarray(peak, dtype=float) if peak is not None else None
    if pk is not None and len(pk) != n:
        raise ValueError("peak must match the length of band_lo.")

    # Default row ordering by the peak centre of cluster mass. barh draws row 0
    # at the bottom and row n-1 at the top, so an ascending sort of the centre
    # puts the innermost H-Locus at the bottom and the outermost at the top —
    # i.e. reading the panel top-to-bottom goes outer -> inner.
    if sort:
        _center = pk.copy() if pk is not None else (lo + hi) / 2.0
        _center = np.where(np.isfinite(_center), _center, -np.inf)
        _order = np.argsort(_center, kind="stable")
        if str(sort).replace("-", "_") == "inner_to_outer":
            _order = _order[::-1]
        lo, hi, is_up = lo[_order], hi[_order], is_up[_order]
        if pk is not None:
            pk = pk[_order]
        if labels is not None:
            labels = [list(labels)[i] for i in _order]

    if ax is None:
        fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 18.0))
        _, ax = plt.subplots(figsize=(5.8, fig_h))

    y = np.arange(n)
    for _yi, _lo, _hi, _up in zip(y, lo, hi, is_up):
        if not (np.isfinite(_lo) and np.isfinite(_hi)):
            continue
        _left, _width = min(_lo, _hi), abs(_hi - _lo)
        _col = up_color if _up else down_color
        ax.barh(_yi, _width, left=_left, height=bar_height,
                color=_col, alpha=alpha, edgecolor=edge_color,
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
    distance_label="physical distance from border (µm)  (<0 tumour | \u2265 0 stroma)",
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


def plot_hloci_bands_bidirectional(
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


# ── Family-consistent naming aliases ──────────────────────────────────────
# The GAM-smoothed H-plot panels are named the "H-GAM Plot" (per-group smooth ±
# CI) and the "H-ΔGAM Plot" (high−low difference ± propagated CI). These
# aliases expose those names while keeping the original ``plot_hplot_gam`` /
# ``plot_delta_hplot_gam`` entry points for backward compatibility.
plot_hgam = plot_hplot_gam
plot_hgam_delta = plot_delta_hplot_gam
plot_delta_hgam = plot_delta_hplot_gam  # legacy alias

# The location-of-the-border-band panel is named the "H-Loci Summary" (each row
# is one feature's H-Locus: position + cluster-mass thickness + direction glyph).
# ``plot_signpost`` is retained as a backward-compatible alias of the earlier
# working name.
plot_signpost = plot_hloci_summary

