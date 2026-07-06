"""Bridge: read an ``hplot-outputs.csv`` back into engine stats.

Lets users re-plot a saved H-Plot result without any AnnData, e.g. when the
scoring pass ran elsewhere and only the CSV was kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# candidate column names -> canonical role
_LAYER = ("layer", "hplot_layer", "signed_layer")
_DIST = ("distance", "distance_um", "hplot_distance_um", "signed_distance_um")
_MEAN = ("mean", "target_type_prop", "value", "proportion")
_LOWER = ("ci_lower", "lower", "ci_low")
_UPPER = ("ci_upper", "upper", "ci_high")
_N = ("n", "all_count", "count", "n_cells")


def _first(cols, candidates):
    lut = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lut:
            return lut[cand]
    return None


def read_hplot_csv(path, group_col=None):
    """Read an H-Plot CSV into ``{group -> stats DataFrame}``.

    The returned DataFrames use the engine schema (``layer``, ``distance``,
    ``mean``, ``ci_lower``, ``ci_upper``, ``n``) so they can be passed straight
    to :func:`hplot.plotting.plot_hplot`. When CI columns are absent the band
    collapses onto the mean line.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    group_col : str | None
        Column that splits rows into multiple curves. When ``None`` every row
        forms a single ``'target'`` curve.
    """
    raw = pd.read_csv(path)
    cols = list(raw.columns)

    layer_c = _first(cols, _LAYER)
    if layer_c is None:
        raise KeyError(f"No layer column found in {path!r}; looked for {_LAYER}.")
    mean_c = _first(cols, _MEAN)
    if mean_c is None:
        raise KeyError(f"No value/mean column found in {path!r}; looked for {_MEAN}.")
    dist_c = _first(cols, _DIST)
    lower_c = _first(cols, _LOWER)
    upper_c = _first(cols, _UPPER)
    n_c = _first(cols, _N)

    def _to_stats(sub):
        out = pd.DataFrame({
            "layer": pd.to_numeric(sub[layer_c], errors="coerce"),
            "mean": pd.to_numeric(sub[mean_c], errors="coerce"),
        })
        out["distance"] = (pd.to_numeric(sub[dist_c], errors="coerce")
                           if dist_c else np.nan)
        out["ci_lower"] = (pd.to_numeric(sub[lower_c], errors="coerce")
                           if lower_c else out["mean"])
        out["ci_upper"] = (pd.to_numeric(sub[upper_c], errors="coerce")
                           if upper_c else out["mean"])
        out["n"] = (pd.to_numeric(sub[n_c], errors="coerce") if n_c else 0)
        return out.sort_values("layer").reset_index(drop=True)

    if group_col and group_col in raw.columns:
        return {str(g): _to_stats(sub) for g, sub in raw.groupby(group_col)}
    return {"target": _to_stats(raw)}
