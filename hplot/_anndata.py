"""AnnData glue for hplot: lazy imports and adata -> tidy-DataFrame extraction.

Keeps every AnnData/squidpy dependency out of the framework-agnostic engine
(``core``/``stats``/``plotting``). Nothing here is imported at package import
time unless the user calls the ``pp``/``tl``/``pl`` API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_anndata():
    """Import anndata or raise a friendly install hint."""
    try:
        import anndata  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The AnnData API of hplot needs anndata. Install it with:\n"
            "    pip install 'hplot[anndata]'   # or  pip install 'hplot[squidpy]'"
        ) from exc


def _gene_values(adata, gene):
    """Dense 1-D expression vector for one gene, sparse-aware."""
    from scipy.sparse import issparse

    if gene not in set(map(str, adata.var_names)):
        raise KeyError(
            f"target={gene!r} is neither a var_name nor handled here; for a "
            "cell-type proportion pass value_kind='proportion' and an .obs column."
        )
    if not adata.var_names.is_unique:
        raise ValueError(
            "adata.var_names are not unique; call adata.var_names_make_unique() "
            f"before selecting target={gene!r}."
        )
    X = adata[:, gene].X
    x = np.asarray(X.todense()).ravel() if issparse(X) else np.asarray(X).ravel()
    return x.astype(float)


def _sample_vector(adata, sample_key):
    if sample_key is None:
        return np.full(adata.n_obs, "sample0", dtype=object)
    return adata.obs[sample_key].astype(str).to_numpy()


def _category_order(adata, col):
    s = adata.obs[col]
    if str(s.dtype) == "category":
        return [str(c) for c in s.cat.categories]
    return sorted(str(c) for c in pd.unique(s.dropna()))


def _adata_to_tidy(adata, target, *, value_kind="expression", groupby=None,
                   sample_key=None, layer_key="hplot_layer",
                   distance_key="hplot_distance_um", zscore=False):
    """Build a long per-(sample, group, layer) DataFrame the engine can fit.

    Returns ``(df, group_order)`` where ``df`` has columns
    ``sample, group, layer, distance, value``. The engine then averages
    ``value`` across samples per layer (``smoother='mean'``), so a single-sample
    AnnData yields the raw per-layer curve.
    """
    obs = adata.obs
    layer = pd.to_numeric(obs[layer_key], errors="coerce").to_numpy()
    if distance_key in obs.columns:
        distance = pd.to_numeric(obs[distance_key], errors="coerce").to_numpy()
    else:
        distance = np.full(adata.n_obs, np.nan)
    sample = _sample_vector(adata, sample_key)
    fin = np.isfinite(layer)

    if value_kind == "proportion":
        if target not in obs.columns:
            raise KeyError(
                f"value_kind='proportion' needs target={target!r} to be an .obs "
                "column of cell categories."
            )
        order = _category_order(adata, target)
        cat = obs[target].astype(str).to_numpy()
        df = pd.DataFrame({"sample": sample[fin], "layer": layer[fin].astype(int),
                           "distance": distance[fin], "cat": cat[fin]})
        rows = []
        for (s, lay), g in df.groupby(["sample", "layer"], sort=True):
            denom = len(g)
            dist_vals = g["distance"].to_numpy()
            dmean = (float(np.nanmean(dist_vals))
                     if np.isfinite(dist_vals).any() else np.nan)
            vc = g["cat"].value_counts()
            for c in order:
                rows.append({"sample": s, "group": c, "layer": int(lay),
                             "distance": dmean, "value": vc.get(c, 0) / denom})
        return pd.DataFrame(rows), order

    # expression mode
    value = _gene_values(adata, target)
    if groupby is not None:
        grp = obs[groupby].astype(str).to_numpy()
        order = _category_order(adata, groupby)
    else:
        grp = np.full(adata.n_obs, "all", dtype=object)
        order = ["all"]

    df = pd.DataFrame({"sample": sample, "layer": layer, "distance": distance,
                       "group": grp, "value": value})[fin]
    df["layer"] = df["layer"].astype(int)

    if zscore:
        def _z(v):
            sd = v.std()
            return (v - v.mean()) / sd if sd > 0 else v * 0.0
        df["value"] = df.groupby("sample")["value"].transform(_z)

    agg = (df.groupby(["sample", "group", "layer"], as_index=False)
             .agg(value=("value", "mean"), distance=("distance", "mean")))
    return agg, order
