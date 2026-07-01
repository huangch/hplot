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
