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
                   distance_key="hplot_distance_um", zscore=False,
                   exclude_base=False, min_base_excluded_count=1,
                   base_cluster_key=None, base_categories=None):
    """Build a long per-(sample, group, layer) DataFrame the engine can fit.

    Returns ``(df, group_order)`` where ``df`` has columns
    ``sample, group, layer, distance, value``. The engine then averages
    ``value`` across samples per layer (``smoother='mean'``), so a single-sample
    AnnData yields the raw per-layer curve.

    When ``exclude_base`` is True (proportion mode only) the per-layer
    denominator counts **non-base** cells only, i.e. the target fraction is taken
    among cells whose ``base_cluster_key`` value is not in ``base_categories``
    (both default to what :func:`hplot.pp.border_layers` recorded in
    ``adata.uns['hplot_border']``). Layers with fewer than
    ``min_base_excluded_count`` non-base cells yield NaN and are dropped.
    """
    obs = adata.obs
    layer = pd.to_numeric(obs[layer_key], errors="coerce").to_numpy()
    if distance_key in obs.columns:
        distance = pd.to_numeric(obs[distance_key], errors="coerce").to_numpy()
    else:
        distance = np.full(adata.n_obs, np.nan)
    sample = _sample_vector(adata, sample_key)
    fin = np.isfinite(layer)

    if value_kind in ("proportion", "fraction"):
        if target not in obs.columns:
            raise KeyError(
                f"value_kind='proportion' needs target={target!r} to be an .obs "
                "column of cell categories."
            )
        order = _category_order(adata, target)
        cat = obs[target].astype(str).to_numpy()

        base_set = set()
        if exclude_base:
            bkey = base_cluster_key
            bcats = base_categories
            if bkey is None or bcats is None:
                info = adata.uns.get("hplot_border", {})
                bkey = bkey or info.get("cluster_key")
                bcats = bcats if bcats is not None else info.get("base_categories")
            if not bkey or not bcats:
                raise KeyError(
                    "exclude_base=True needs the base region; run "
                    "hplot.pp.border_layers first, or pass base_cluster_key and "
                    "base_categories explicitly."
                )
            if bkey not in obs.columns:
                raise KeyError(f"base_cluster_key={bkey!r} not in adata.obs.")
            base_set = {str(c) for c in bcats}
            is_base = obs[bkey].astype(str).isin(base_set).to_numpy()

        cols = {"sample": sample[fin], "layer": layer[fin].astype(int),
                "distance": distance[fin], "cat": cat[fin]}
        if exclude_base:
            cols["is_base"] = is_base[fin]
        df = pd.DataFrame(cols)
        # A base category has no meaning as a target once the base is excluded.
        out_order = [c for c in order if not (exclude_base and str(c) in base_set)]
        rows = []
        for (s, lay), g in df.groupby(["sample", "layer"], sort=True):
            denom = int((~g["is_base"]).sum()) if exclude_base else len(g)
            dist_vals = g["distance"].to_numpy()
            dmean = (float(np.nanmean(dist_vals))
                     if np.isfinite(dist_vals).any() else np.nan)
            vc = g["cat"].value_counts()
            too_few = exclude_base and denom < min_base_excluded_count
            for c in out_order:
                val = np.nan if (too_few or denom == 0) else vc.get(c, 0) / denom
                rows.append({"sample": s, "group": c, "layer": int(lay),
                             "distance": dmean, "value": val})
        return pd.DataFrame(rows), out_order

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
