import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_hplot(grouped_stats, distance_col=None, distance_unit=None, ci_show=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for label, df in grouped_stats.items():
        x = df['distance'] if distance_col else df['layer']
        y = df["mean"]
        ax.plot(x, y, label=str(label), drawstyle="steps-post")

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.3, step="post")

    ax.set_xlabel(f"Distance from tumor boundary ({'distance' if distance_col else 'layer index'}{'in '+distance_unit if distance_unit else ''})")  
    ax.set_ylabel("Proportion of target cell type")
    ax.set_title("Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax