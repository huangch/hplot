import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_hplot(grouped_stats, distance_unit=None, ci_show=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for label, df in grouped_stats.items():
        x = df['layer']
        y = df["mean"]
        ax.plot(x, y, label=str(label), drawstyle="steps-post")

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.3, step="post")

    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

    def distance_formattyer(val, pos):
        dst_list = []
        for label, df in grouped_stats.items():
            if val in df['layer']:
                dst_list.append(df[df['layer']==val, 'distance'].values[0])
        dst_mean = np.mean(dst_list)
        return f"{val:g}\n{dst_mean:.3f}" 
    
    ax.xaxis.set_major_formatter(FuncFormatter(distance_formattyer))
    ax.set_xlabel(f"Distance to tumor boundary\nCellular layers / Euclidean distance{' ('+distance_unit+')' if distance_unit else ''})")  
    ax.set_ylabel("Proportion of lymphocytes")
    ax.set_title("Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax