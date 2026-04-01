import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_hplot(
   target_grouped_stats,
   base_grouped_stats=None,
   distance_unit=None,
   ci_show=True,
   ax=None,
   display_base_type="tumor",
   display_target_type="immune cells",
   color_map=None,
   palette=None,
   legend_order=None,
   legend_title="Group",
   legend_kwargs=None,
):
   """
   Plot H-plot curves from precomputed grouped_stats.
   Parameters
   ----------
   target_grouped_stats : dict[str, pd.DataFrame]
       Mapping from group label -> stats DataFrame.
       DataFrame must contain columns: 'layer', 'mean' and (if ci_show) 'ci_lower', 'ci_upper'.
       If distance tick labels are desired, DataFrame should contain column: 'distance'.
   distance_unit : str | None
       Unit string shown on x tick second line (optional).
   ci_show : bool
       Whether to draw confidence interval bands using fill_between.
   ax : matplotlib.axes.Axes | None
       Existing axis to draw into; if None, create a new figure/axis.
   display_base_type : str
       Used only for title text.
   display_target_type : str
       Used only for y-axis label text.
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
   """
   if legend_kwargs is None:
       legend_kwargs = {}
   # default palette if neither specified
   if color_map is None and palette is None:
       palette = plt.cm.tab10.colors
   if ax is None:
       _, ax = plt.subplots(figsize=(6, 4))
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
           drawstyle="steps-post",
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
               step="post",
           )
   # Plot base proportion lines if provided
   if base_grouped_stats:
       for i, (label, df) in enumerate(base_grouped_stats.items()):
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
               label=f"{label} (base)",
               color=color,
               drawstyle="steps-post",
               linewidth=2,
           )
           if ci_show:
               if ("ci_lower" not in df.columns) or ("ci_upper" not in df.columns):
                   raise ValueError(
                       f"ci_show=True but '{label}' base stats missing ci_lower/ci_upper columns."
                   )
               ax.fill_between(
                   x,
                   df["ci_lower"].to_numpy(),
                   df["ci_upper"].to_numpy(),
                   color=color,
                   alpha=0.25,
                   step="post",
               )
   ax.ticklabel_format(axis="x", style="plain", useOffset=False)
   # Optional: show layer + mean distance per layer on x ticks
   def distance_formatter(value, _pos):
        """Formatter that appends the mean physical distance for a layer."""

        layer_index = int(round(value))
        distances = []
        for stats_df in target_grouped_stats.values():
            if "distance" not in stats_df.columns:
                continue
            mask = stats_df["layer"].round().astype(np.int32) == layer_index
            if mask.any():
                distances.append(stats_df.loc[mask, "distance"].dropna().mean())

        if distances:
            distance_value = float(np.mean(distances))
            if np.isnan(distance_value):
                return f"{layer_index:g}"
            # if distance_unit:
            #     return f"{layer_index:g}\n{distance_value:.1f} {distance_unit}"
            return f"{layer_index:g}\n{distance_value:.1f}"
        return f"{layer_index:g}"
         
   ax.xaxis.set_major_formatter(FuncFormatter(distance_formatter))
   ax.set_xlabel(
       "Layerwise cellular distance from"
       f" {display_base_type} border"
       + (
           f"\nPhysical distance ({distance_unit}) from {display_base_type} border"
           if distance_unit
           else ""
       )
   )

   ax.set_ylabel(f"Proportion of {display_target_type}")
   ax.set_title(f"{display_base_type.capitalize()} Spatial Heterogeneity Profile (H-plot)")
   ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   ax.xaxis.set_major_formatter(FuncFormatter(distance_formatter))
   ax.grid(True, linestyle="--", alpha=0.5)
   ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
   # Legend ordering
   handles, labels = ax.get_legend_handles_labels()
   if legend_order is not None:
       idx = [labels.index(l) for l in legend_order if l in labels]
       handles = [handles[i] for i in idx]
       labels = [labels[i] for i in idx]
   ax.legend(handles, labels, title=legend_title, **legend_kwargs)
   return ax
    
def plot_hplotx(grouped_stats, distance_unit=None, ci_show=True, ax=None, display_base_type='tumor', display_target_type='immune cells', color_map=None, palette=None, legend_order=None, legend_title="Group", legend_kwargs=None,):

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

        ax.plot(x, y, label=str(label), color=color, drawstyle="steps-post")

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], color=color, alpha=0.3, step="post")

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
    ax.set_xlabel(f"Layerwise cellular distance from {display_base_type} border\n(Physical distance{' ('+distance_unit+') ' if distance_unit else ' '}from {display_base_type} border)")  
    ax.set_ylabel(f"Proportion of {display_target_type}")
    ax.set_title("Tumor Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax
