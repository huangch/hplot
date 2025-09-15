import pandas as pd
import numpy as np
from scipy.stats import t, norm

def compute_layer_stats(df, value_col, layer_col, distance_col, ci=0.95):
    grouped = df.groupby(layer_col)
    summary = []

    for layer, group in grouped:
        values = group[value_col].values
        n = len(values)

        if n > 1:
            distance = group[distance_col].mean() if distance_col else None
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