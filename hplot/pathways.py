"""Pathway/signature scoring on the border ruler.

The H-Pathway question is *where a program sits relative to the boundary*. This
module supplies the measurement half of that answer and nothing else:

- :func:`ucell_scores`          per-cell rank-based signature score (UCell);
- :func:`pathway_layer_profile` per-layer mean score for many signatures at once;
- :func:`pathway_layer_profile_h5ad` the same, starting from an ``.h5ad`` on disk.

None of these test a hypothesis. Scoring is deliberately separated from
inference because the two failed together in an earlier design: a self-contained
"does this set depart from its own baseline" test is close to preordained on a
targeted panel and made pathway *names* uninterpretable. Use
:func:`hplot.hpathway_layer_ora` (competitive, per layer) when the question is
*which* sets are border-organised, and :func:`hplot.hpathway_score_grid` when the
set list is already fixed and only its position is at issue.

Signature scores carry no elevated/depressed direction: a gene set mixes
positively and negatively regulated members, so the mean of its members is not a
signed statement about pathway activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_dense(block):
    return np.asarray(block.todense() if hasattr(block, "todense") else block)


def ucell_scores(X, sig_idx, *, max_rank=1500, chunk=20000):
    """Per-cell UCell score for each signature.

    UCell ranks the genes within each cell and scores a signature by the
    Mann-Whitney U of its members' ranks, so the score is invariant to
    per-cell sequencing depth and to any monotone transform of expression --
    which is what makes it comparable across the layers of a border ruler.

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
    per row of ``X``) from :func:`hplot.border_layers_from_coords` or from
    ``hplot.pp.border_layers``, so the layering is unchanged and only the
    scoring/aggregation is packaged.

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
        Extra scalar columns to attach to every output row (e.g. an arm label).

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
        block = np.asarray(_to_dense(Xk[start:stop]), dtype=np.float32)
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


def pathway_layer_profile_h5ad(path, signatures, *, base_col,
                               h5ad_name="annotated.h5ad", spatial_key="spatial",
                               k=2, n_min=10, ratio=0.2, max_edge=25.0,
                               sample=None, extra=None, cell_mask=None,
                               max_rank=1500, chunk=8000, cache_path=None,
                               force=False):
    """Border-layer a run from disk and return its per-layer signature profile.

    Convenience wrapper: reads the ``.h5ad``, derives the signed border layer
    with :func:`hplot.border_layers_from_coords`, then calls
    :func:`pathway_layer_profile`. Results can be cached per run because the
    layering is deterministic given the same parameters.

    Parameters
    ----------
    path : str
        Either the ``.h5ad`` file or the run directory containing ``h5ad_name``.
    signatures : dict[str, sequence[str]]
        Signature name -> gene symbols.
    base_col : str
        Boolean ``.obs`` column marking the base (e.g. malignant) compartment.
    cell_mask : callable | None
        Optional ``adata -> boolean array`` restricting which cells are scored,
        e.g. to score a pathway only inside the population it was selected in.
        Cells outside the mask are dropped before layering statistics are taken.
    cache_path : str | None
        Joblib cache for the returned frame.

    Returns
    -------
    pandas.DataFrame
        As :func:`pathway_layer_profile`, or ``None`` when the mask leaves no
        cell with a finite layer.
    """
    import os

    import joblib

    if cache_path and (not force) and os.path.exists(cache_path):
        return joblib.load(cache_path)

    import anndata as ad

    from ._geometry import border_layers_from_coords

    fp = path if os.path.isfile(path) else os.path.join(path, h5ad_name)
    if sample is None:
        sample = os.path.basename(os.path.dirname(fp)) or os.path.basename(fp)
    adata = ad.read_h5ad(fp)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    is_base = np.asarray(adata.obs[base_col]).astype(bool)
    _um, signed = border_layers_from_coords(coords, is_base, A=None, k=k, n_min=n_min,
                                            ratio=ratio, max_edge=max_edge)
    keep = np.isfinite(signed)
    if cell_mask is not None:
        keep = keep & np.asarray(cell_mask(adata)).astype(bool)
    prof = None
    if keep.any():
        prof = pathway_layer_profile(
            adata.X[keep], signed[keep].astype(int), signatures,
            var_names=adata.var_names, sample=sample,
            max_rank=max_rank, chunk=chunk, extra=extra)
    del adata
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        joblib.dump(prof, cache_path, compress=3)
    return prof
