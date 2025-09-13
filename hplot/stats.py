import pandas as pd
import numpy as np
from scipy.stats import t, norm

def compute_layer_stats(df, value_col, layer_col, region_col, distance_col, ci=0.95):
    grouped = df.groupby(layer_col)
    summary = []

    for layer, group in grouped:
        distance = group[distance_col].mean() if distance_col else None
        values = group[value_col].values
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)

        if n > 30:
            z = norm.ppf(1 - (1 - ci) / 2)
        else:
            z = t.ppf(1 - (1 - ci) / 2, df=n - 1)

        ci_lower = mean - z * sem
        ci_upper = mean + z * sem

        summary.append({
            'layer': layer,
            'distance': distance,
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n
        })

    return pd.DataFrame(summary)