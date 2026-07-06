"""Preprocessing: assign each cell a graph-geodesic border layer.

Mirrors the ``scanpy`` ``pp`` convention: mutate ``adata`` in place, write
per-cell results to ``.obs`` and the run parameters to ``.uns``.

    import squidpy as sq
    import hplot
    sq.gr.spatial_neighbors(adata)                       # optional
    hplot.pp.border_layers(adata, "cell_type", ["tumour"])
"""

from __future__ import annotations

import warnings

import numpy as np

from ._anndata import _require_anndata, _sample_vector
from ._geometry import border_layers_from_coords


def border_layers(
    adata,
    cluster_key,
    base_categories,
    *,
    spatial_key="spatial",
    connectivity_key="spatial_connectivities",
    sample_key=None,
    k=2,
    n_min=10,
    ratio=0.2,
    max_edge=25.0,
    build_graph_if_missing=True,
    layer_key="hplot_layer",
    distance_key="hplot_distance_um",
    copy=False,
):
    """Assign a signed border layer + micron distance to every cell.

    Also available as :func:`hplot.gr.border_layers` (squidpy-style alias) —
    both names refer to the same function.

    Graph source (both, with fallback): if ``adata.obsp[connectivity_key]``
    exists it is used as the spatial graph; otherwise a Delaunay graph is built
    from ``adata.obsm[spatial_key]`` (pruned at ``max_edge``) when
    ``build_graph_if_missing`` is True.

    Parameters
    ----------
    adata : AnnData
    cluster_key : str
        ``.obs`` column defining cell compartments.
    base_categories : str | sequence[str]
        Value(s) of ``cluster_key`` that make up the base (e.g. tumour) region.
    sample_key : str | None
        ``.obs`` column identifying independent tissues; the border graph is
        computed per sample so hops never cross samples.
    k, n_min, ratio, max_edge : see :func:`hplot._geometry.border_layers_from_coords`.
    build_graph_if_missing : bool
        Build a Delaunay graph when no precomputed graph is present.
    layer_key, distance_key : str
        ``.obs`` columns written with the signed hop layer and signed microns.
    copy : bool
        Return a modified copy instead of writing in place.

    Returns
    -------
    AnnData | None
        The modified AnnData when ``copy=True``, else ``None``.
    """
    _require_anndata()
    adata = adata.copy() if copy else adata

    if isinstance(base_categories, str):
        base_categories = [base_categories]
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key={cluster_key!r} not in adata.obs.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"spatial_key={spatial_key!r} not in adata.obsm.")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    is_base = adata.obs[cluster_key].astype(str).isin(
        [str(c) for c in base_categories]).to_numpy()
    sample = _sample_vector(adata, sample_key)

    have_graph = connectivity_key in adata.obsp
    if not have_graph and not build_graph_if_missing:
        raise KeyError(
            f"No adata.obsp[{connectivity_key!r}] and build_graph_if_missing=False; "
            "run sq.gr.spatial_neighbors(adata) first or allow Delaunay fallback."
        )
    A_full = adata.obsp[connectivity_key] if have_graph else None

    signed_um = np.full(adata.n_obs, np.nan, dtype=float)
    signed_hops = np.full(adata.n_obs, np.nan, dtype=float)

    for s in np.unique(sample):
        idx = np.where(sample == s)[0]
        if idx.size < 4:
            warnings.warn(
                f"Sample {s!r} has {idx.size} cells (< 4); border layers left NaN.",
                stacklevel=2,
            )
            continue
        A_sub = A_full[idx][:, idx] if A_full is not None else None
        try:
            um, hops = border_layers_from_coords(
                coords[idx], is_base[idx], A=A_sub, k=k, n_min=n_min,
                ratio=ratio, max_edge=max_edge,
            )
        except Exception as exc:  # e.g. QhullError on collinear/degenerate coords
            warnings.warn(
                f"Border-layer computation failed for sample {s!r} "
                f"({type(exc).__name__}: {exc}); left NaN.",
                stacklevel=2,
            )
            continue
        signed_um[idx] = um
        signed_hops[idx] = hops

    adata.obs[layer_key] = signed_hops
    adata.obs[distance_key] = signed_um
    adata.uns["hplot_border"] = {
        "cluster_key": str(cluster_key),
        "base_categories": [str(c) for c in base_categories],
        "graph_source": "precomputed" if A_full is not None else "delaunay",
        "connectivity_key": str(connectivity_key),
        "spatial_key": str(spatial_key),
        "sample_key": "" if sample_key is None else str(sample_key),
        "k": int(k),
        "n_min": int(n_min),
        "ratio": float(ratio),
        "max_edge": float(max_edge),
        "layer_key": str(layer_key),
        "distance_key": str(distance_key),
        "n_border_layers": int(np.unique(signed_hops[np.isfinite(signed_hops)]).size),
    }
    return adata if copy else None
