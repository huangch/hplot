import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_hplot(grouped_stats, distance_unit=None, ci_show=True, ax=None, display_base_type='tumor', display_target_type='immune cells'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    plt.tight_layout()
    
    for label, df in grouped_stats.items():
        x = df['layer'].round().astype(np.int32)
        y = df["mean"]
        ax.plot(x, y, label=str(label), drawstyle="steps-post")

        if ci_show:
            ax.fill_between(x, df["ci_lower"], df["ci_upper"], alpha=0.3, step="post")

    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

    def distance_formattyer(val, pos):
        dst_list = []
        for _, df in grouped_stats.items():
            if int(round(val)) in df['layer'].round().astype(np.int32).tolist():
                dst = df[df['layer'].round().astype(np.int32)==int(round(val))]['distance'].mean()
                dst_list.append(dst)
        
        if len(dst_list) > 0:
            dst_mean = np.mean(dst_list)
            tick_label = f"{val:g}\n{dst_mean:.1f}"
        else:
            tick_label = f"{val:g}\n"
            
        return tick_label

    ax.xaxis.set_major_formatter(FuncFormatter(distance_formattyer))
    ax.set_xlabel(f"Layerwise cellular distance from {display_base_type} border\nPhysical distance{' ('+distance_unit+') ' if distance_unit else ' '}from tumor border")  
    ax.set_ylabel(f"Proportion of {display_target_type}")
    ax.set_title("Tumor Spatial Heterogeneity Profile (H-Plot)")
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Highlight layer 0 (tumor boundary)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

    # Force integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax