"""Tools: fit an H-Plot from AnnData and stash the result in ``adata.uns``.

Mirrors ``scanpy.tl``: read from ``adata``, run the method, write results back
to ``adata`` (here ``adata.uns['hplot']`` as an h5ad-safe dict)::

    hplot.pp.border_layers(adata, "cell_type", ["tumour"])
    hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype",
                   value_kind="expression")
    hplot.pl.hplot(adata)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._anndata import _adata_to_tidy, _require_anndata
from ._serial import serialize
from .core import HPlot


def hplot(
    adata,
    target,
    *,
    value_kind="expression",
    groupby=None,
    sample_key=None,
    layer_key="hplot_layer",
    distance_key="hplot_distance_um",
    smoother="mean",
    zscore=False,
    exclude_base=False,
    min_base_excluded_count=1,
    base_cluster_key=None,
    base_categories=None,
    color_map=None,
    legend_title=None,
    display_base_type="tumour border",
    display_target_type=None,
    key_added="hplot",
    copy=False,
    **fit_kwargs,
):
    """Fit an H-Plot from an AnnData carrying border layers.

    Requires a prior :func:`hplot.pp.border_layers` run (or matching
    ``layer_key``/``distance_key`` columns in ``adata.obs``).

    Parameters
    ----------
    target : str
        A ``var_name`` (gene) when ``value_kind='expression'`` /
        ``'interaction'``; an ``.obs`` categorical column when
        ``value_kind='proportion'`` / ``'fraction'``.
    value_kind : {'expression', 'proportion', 'fraction', 'interaction'}
        Quantity profiled; also selects the y-axis phrasing at plot time.
    groupby : str | None
        ``.obs`` column splitting cells into one curve each (expression modes).
    sample_key : str | None
        Replicate/tissue column; curves are averaged across samples per layer.
    zscore : bool
        Z-score the gene per sample before aggregating (expression modes).
    key_added : str
        ``adata.uns`` key to write the serialised result to.
    copy : bool
        Operate on and return a copy instead of writing in place.

    Returns
    -------
    AnnData | None
        The modified AnnData when ``copy=True``, else ``None``.
    """
    _require_anndata()
    adata = adata.copy() if copy else adata

    if layer_key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs[{layer_key!r}] missing; run hplot.pp.border_layers first."
        )

    is_proportion = value_kind in ("proportion", "fraction")
    if exclude_base and not is_proportion:
        import warnings
        warnings.warn(
            f"exclude_base=True is ignored for value_kind={value_kind!r}; it "
            "only affects proportion/fraction denominators.",
            stacklevel=2,
        )
    df, group_order = _adata_to_tidy(
        adata, target, value_kind=value_kind,
        groupby=None if is_proportion else groupby,
        sample_key=sample_key, layer_key=layer_key,
        distance_key=distance_key, zscore=zscore,
        exclude_base=exclude_base and is_proportion,
        min_base_excluded_count=min_base_excluded_count,
        base_cluster_key=base_cluster_key, base_categories=base_categories,
    )

    use_group = is_proportion or (groupby is not None)
    if display_target_type is None:
        display_target_type = target
    if legend_title is None:
        legend_title = (str(target) if is_proportion
                        else (str(groupby) if groupby else str(target)))

    hp = HPlot()
    hp.fit(
        df,
        targets="value",
        layer="layer",
        group="group" if use_group else None,
        distance="distance",
        unit="um",
        color_map=color_map,
        legend_order=group_order if use_group else None,
        legend_title=legend_title,
        smoother=smoother,
        **fit_kwargs,
    )

    adata.uns[key_added] = serialize(
        hp, value_kind=value_kind, display_base_type=display_base_type,
        display_target_type=display_target_type, target=target,
        group_order=group_order if use_group else None,
        color_map=color_map, legend_title=legend_title,
    )
    return adata if copy else None


# ---------------------------------------------------------------------------
# UCell pathway/signature scoring for the H-Pathway Summary
# ---------------------------------------------------------------------------
def ucell_scores(X, sig_idx, *, max_rank=1500, chunk=20000):
    """Per-cell UCell scores for each signature (rank-based, bounded ``[0, 1]``).

    The per-cell rank matrix is computed once per cell-chunk and reused for
    every signature in ``sig_idx``, so adding signatures is cheap; peak memory
    is bounded by ``chunk``.

    Parameters
    ----------
    X : scipy.sparse matrix | ndarray, shape (n_cells, n_genes)
        Expression matrix (rows = cells).
    sig_idx : dict[str, ndarray[int]]
        Signature name -> integer column indices into ``X`` of its genes.
    max_rank : int
        UCell rank cap; genes ranked beyond this contribute the capped rank.
    chunk : int
        Number of cells scored per block.

    Returns
    -------
    dict[str, ndarray[float32]]
        Signature name -> per-cell score (length ``n_cells``).
    """
    n = X.shape[0]
    names = list(sig_idx)
    out = {nm: np.empty(n, dtype=np.float32) for nm in names}
    mr = int(max_rank)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.asarray(_to_dense(X[start:stop]), dtype=np.float32)
        b, g = block.shape
        mr_eff = max(1, min(mr, g))
        order = np.argsort(-block, axis=1, kind="stable")
        ranks = np.empty((b, g), dtype=np.int32)
        rows = np.arange(b)[:, None]
        ranks[rows, order] = np.arange(1, g + 1, dtype=np.int32)[None, :]
        np.minimum(ranks, mr_eff + 1, out=ranks)
        for nm in names:
            idx = np.asarray(sig_idx[nm], dtype=np.int64)
            k = int(idx.size)
            if k == 0:
                out[nm][start:stop] = np.nan
                continue
            rsum = ranks[:, idx].sum(axis=1).astype(np.float64)
            U = rsum - k * (k + 1) / 2.0
            out[nm][start:stop] = (1.0 - U / (k * mr_eff)).astype(np.float32)
    return out


def pathway_layer_profile(X, layers, signatures, *, var_names, sample=None,
                          max_rank=1500, chunk=8000, extra=None):
    """Per-layer mean UCell profile for many signatures at once.

    Scales to large catalogs: the per-cell rank matrix is computed once per
    cell-chunk and all signatures are scored in a single sparse membership
    matmul (``ranks @ M``); per-layer means are accumulated on the fly, so
    nothing of size ``(n_cells x n_signatures)`` is ever materialised.

    Geometry is *not* computed here: pass ``layers`` (one integer border layer
    per row of ``X``) from your existing H-Plot pipeline, so the scientific
    layering is unchanged and only the scoring/aggregation is packaged.

    Parameters
    ----------
    X : scipy.sparse matrix | ndarray, shape (n_cells, n_genes)
        Expression for the cells to profile (already restricted to valid layers).
    layers : array-like[int], shape (n_cells,)
        Integer border layer for each row of ``X``.
    signatures : dict[str, sequence[str]]
        Signature name -> gene symbols.
    var_names : sequence[str]
        Gene symbols matching the columns of ``X``.
    sample : hashable | None
        If given, attached as a ``"sample"`` column on every output row.
    max_rank : int
        UCell rank cap.
    chunk : int
        Cells scored per block.
    extra : dict | None
        Extra scalar columns to attach to every output row (e.g. status).

    Returns
    -------
    pandas.DataFrame
        One row per layer with columns ``layer``, one per signature (mean
        per-cell UCell score), ``n_cells``, plus ``sample``/``extra`` if given.
    """
    import scipy.sparse as sp

    L = np.asarray(layers)
    L = L[np.isfinite(L.astype(float))].astype(int) if L.dtype.kind == "f" else L.astype(int)
    if L.shape[0] != X.shape[0]:
        raise ValueError(
            f"layers length {L.shape[0]} != X rows {X.shape[0]}; filter both "
            "to the same valid-layer cells before calling.")

    names = list(signatures)
    nS, G = len(names), X.shape[1]
    var_pos = {str(g): i for i, g in enumerate(var_names)}

    rows, cols = [], []
    kcount = np.zeros(nS, dtype=np.float64)
    for j, nm in enumerate(names):
        gi = [var_pos[str(g)] for g in signatures[nm] if str(g) in var_pos]
        rows.extend(gi)
        cols.extend([j] * len(gi))
        kcount[j] = len(gi)
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(G, nS))

    Xk = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    nk = Xk.shape[0]

    uniq = np.array(sorted(set(L.tolist())), dtype=int)
    lpos = {int(v): i for i, v in enumerate(uniq)}
    lidx = np.array([lpos[int(v)] for v in L], dtype=np.int64)
    nL = len(uniq)
    lay_sum = np.zeros((nL, nS), dtype=np.float64)
    lay_cnt = np.zeros(nL, dtype=np.int64)
    mr = int(max_rank)

    for start in range(0, nk, chunk):
        stop = min(start + chunk, nk)
        block = np.asarray(Xk[start:stop].todense(), dtype=np.float32)
        b, g = block.shape
        mr_eff = max(1, min(mr, g))
        order = np.argsort(-block, axis=1, kind="stable")
        ranks = np.empty((b, g), dtype=np.float32)
        rr = np.arange(b)[:, None]
        ranks[rr, order] = np.arange(1, g + 1, dtype=np.float32)[None, :]
        np.minimum(ranks, mr_eff + 1, out=ranks)
        sumranks = np.asarray(M.T.dot(ranks.T)).T            # (b x nS)
        sc_blk = 1.0 - (sumranks - kcount * (kcount + 1) / 2.0) / (kcount * mr_eff)
        np.add.at(lay_sum, lidx[start:stop], sc_blk.astype(np.float64))
        lay_cnt += np.bincount(lidx[start:stop], minlength=nL)
        del block, order, ranks, sumranks, sc_blk

    lay_mean = lay_sum / np.maximum(lay_cnt, 1)[:, None]
    prof = pd.DataFrame(lay_mean, columns=names)
    prof.insert(0, "layer", uniq)
    prof["n_cells"] = lay_cnt
    if sample is not None:
        prof["sample"] = sample
    if extra:
        for k, v in extra.items():
            prof[k] = v
    return prof


def pathway_layer_profile_adata(
    adata,
    signatures,
    *,
    base_col,
    spatial_key="spatial",
    k=2,
    n_min=10,
    ratio=0.2,
    max_edge=25.0,
    sample=None,
    extra=None,
    max_rank=1500,
    chunk=8000,
    return_layers=False,
):
    """Border-layer H-Pathway profile straight from an AnnData.

    One-call convenience that fuses the two H-Plot steps so callers don't have
    to re-glue them every time:

    1. geometry -- :func:`hplot.border_layers_from_coords` on
       ``adata.obsm[spatial_key]`` and the boolean base mask
       ``adata.obs[base_col]`` (Delaunay -> k-hop -> base region -> border ->
       signed hop layer);
    2. scoring -- :func:`pathway_layer_profile` (per-cell UCell scoring +
       per-layer mean) on the cells that fall on a finite layer.

    File I/O and caching are intentionally left to the caller (they are
    application-specific); everything scientific is packaged here.

    Parameters
    ----------
    adata : AnnData
        Must carry ``obsm[spatial_key]`` (n_cells x 2 coordinates) and a
        boolean-like ``obs[base_col]`` marking the base (e.g. tumour) cells.
    signatures : dict[str, sequence[str]]
        Signature name -> gene symbols (matched against ``adata.var_names``).
    base_col : str
        ``.obs`` column giving the base-region membership per cell.
    spatial_key : str
        ``.obsm`` key with the 2-D coordinates. Default ``"spatial"``.
    k, n_min, ratio, max_edge : see :func:`hplot.border_layers_from_coords`.
        ``max_edge`` is in the same units as the coordinates (multiply by MPP
        if your coordinates are in pixels and you want a micron threshold).
    sample : hashable | None
        Attached as a ``"sample"`` column on every output row.
    extra : dict | None
        Extra scalar columns to attach to every output row (e.g. status).
    max_rank, chunk : see :func:`pathway_layer_profile`.
    return_layers : bool
        If True, also return the per-cell signed layer array (length n_cells,
        ``nan`` where unreachable) for downstream reuse.

    Returns
    -------
    pandas.DataFrame  (or ``(DataFrame, ndarray)`` when ``return_layers=True``)
        One row per layer, as in :func:`pathway_layer_profile`.
    """
    from ._geometry import border_layers_from_coords

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    is_base = np.asarray(adata.obs[base_col]).astype(bool)
    _um, signed = border_layers_from_coords(
        coords, is_base, A=None, k=k, n_min=n_min, ratio=ratio, max_edge=max_edge
    )
    keep = np.isfinite(signed)
    layers = signed[keep].astype(int)
    prof = pathway_layer_profile(
        adata.X[keep], layers, signatures,
        var_names=adata.var_names, sample=sample,
        max_rank=max_rank, chunk=chunk, extra=extra,
    )
    if return_layers:
        return prof, signed
    return prof


def pathway_layer_profile_h5ad(
    path,
    signatures,
    *,
    base_col,
    h5ad_name="annotated.h5ad",
    spatial_key="spatial",
    k=2,
    n_min=10,
    ratio=0.2,
    max_edge=25.0,
    sample=None,
    extra=None,
    max_rank=1500,
    chunk=8000,
    cache_path=None,
    force=False,
):
    """Read an ``.h5ad`` from disk and return its border-layer pathway profile.

    Thin file/cache adapter over :func:`pathway_layer_profile_adata`: it reads
    the slide, runs the full H-Plot geometry + UCell scoring, and (optionally)
    joblib-caches the result. The only thing left for the caller is supplying
    dataset-specific ``extra`` labels (e.g. clinical status), which the package
    cannot know.

    Parameters
    ----------
    path : str
        Either an ``.h5ad`` file, or a directory containing ``h5ad_name``.
    h5ad_name : str
        File name to read when ``path`` is a directory. Default
        ``"annotated.h5ad"``.
    sample : hashable | None
        Slide id attached to every row; defaults to the containing folder name.
    cache_path : str | None
        If given, the profile is loaded from here when present (unless
        ``force``) and written here after computing.
    force : bool
        Recompute and overwrite the cache even if it exists.

    Other parameters are forwarded to :func:`pathway_layer_profile_adata`.

    Returns
    -------
    pandas.DataFrame
        One row per layer, as in :func:`pathway_layer_profile`.
    """
    import os

    if cache_path and (not force) and os.path.exists(cache_path):
        import joblib
        return joblib.load(cache_path)

    import anndata as ad

    fp = path if os.path.isfile(path) else os.path.join(path, h5ad_name)
    if sample is None:
        sample = os.path.basename(os.path.dirname(fp)) or os.path.basename(fp)
    adata = ad.read_h5ad(fp)
    prof = pathway_layer_profile_adata(
        adata, signatures, base_col=base_col, spatial_key=spatial_key,
        k=k, n_min=n_min, ratio=ratio, max_edge=max_edge,
        sample=sample, extra=extra, max_rank=max_rank, chunk=chunk,
    )
    del adata
    if cache_path:
        import joblib
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        joblib.dump(prof, cache_path, compress=3)
    return prof


def _to_dense(block):
    """Densify a sparse row-block; pass dense arrays through unchanged."""
    todense = getattr(block, "todense", None)
    return todense() if todense is not None else block
