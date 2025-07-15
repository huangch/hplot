import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_hplot(grouped_stats, ci_show=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for label, df in grouped_stats.items():
        x = df[df.columns[0]]
        y = df["mean"]
        ax.plot(x, y, label=str(label), drawstyle="steps-post")

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.3, step="post")

    ax.set_xlabel("Distance from tumor boundary (layer index)")
    ax.set_ylabel("Proportion of target cell type")
    ax.set_title("Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax