"""Pure-geometry helpers for graph-geodesic border layers.

Ported from ``sptxinsight.insightlib.insight_helpers`` but stripped of the
WSInsight I/O plumbing (URIPath, HDF5 caches, WSI stubs). Everything here works
on plain numpy/scipy/pandas so ``hplot`` can compute border layers directly
from cell coordinates or a precomputed spatial graph, with no heavy
dependencies.

The public entry point is :func:`border_layers_from_coords`, which returns the
signed graph-hop layer and the signed micron distance to the nearest border
cell for every cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Spatial graph
# ---------------------------------------------------------------------------

def delaunay_edges(points, max_edge=None):
    """Delaunay triangulation edges, optionally pruned by length.

    Parameters
    ----------
    points : (N, 2) array-like
        Cell centre coordinates.
    max_edge : float | None
        Drop edges longer than this. ``None`` keeps every edge.

    Returns
    -------
    pandas.DataFrame
        Columns ``source``, ``target`` (int) and ``length`` (float), one row
        per unique undirected edge.
    """
    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=np.float64)
    tri = Delaunay(points)
    simplices = tri.simplices.astype(np.int64)
    pairs = np.concatenate(
        [simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [1, 2]]], axis=0
    )
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    length = np.linalg.norm(points[src] - points[dst], axis=1)
    if max_edge is not None:
        keep = length < max_edge
        src, dst, length = src[keep], dst[keep], length[keep]
    return pd.DataFrame({"source": src, "target": dst, "length": length})


def adjacency_from_edges(n, edges):
    """Symmetric 0/1 CSR adjacency (no self-loops) from an edge table."""
    from scipy.sparse import csr_matrix

    if n == 0:
        return csr_matrix((0, 0), dtype=np.uint8)
    if len(edges) == 0:
        return csr_matrix((n, n), dtype=np.uint8)
    s = edges["source"].to_numpy(dtype=np.int64)
    t = edges["target"].to_numpy(dtype=np.int64)
    rows = np.concatenate([s, t])
    cols = np.concatenate([t, s])
    A = csr_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)),
                   shape=(n, n), dtype=np.uint8)
    A.data[:] = 1
    return A


def binarize_adjacency(conn):
    """Coerce a (possibly weighted) sparse connectivity matrix to 0/1 CSR."""
    from scipy.sparse import csr_matrix

    A = csr_matrix(conn, dtype=np.uint8).copy()
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _khop_reachability(A, k):
    """``(A + I)^k`` binarised: k-hop reachability with self-loops."""
    from scipy.sparse import eye

    n = A.shape[0]
    M = (A + eye(n, dtype=np.uint8, format="csr")).tocsr()
    M.data[:] = 1
    Mk = M
    for _ in range(max(k - 1, 0)):
        Mk = (Mk @ M).tocsr()
        Mk.data[:] = 1
    return Mk


# ---------------------------------------------------------------------------
# Base region / border / distance
# ---------------------------------------------------------------------------

def base_region_mask(Mk, is_base, n_min, ratio):
    """Cells whose k-hop neighbourhood is large enough and base-enriched."""
    is_base = np.asarray(is_base, dtype=np.float32)
    ones = np.ones_like(is_base)
    b = np.asarray(Mk @ is_base).ravel()
    nn = np.asarray(Mk @ ones).ravel()
    safe = np.maximum(nn, 1.0)
    return (nn >= n_min) & (b / safe >= ratio)


def border_mask(A, is_region):
    """Base-region cells that have at least one non-base-region neighbour."""
    is_region = np.asarray(is_region, dtype=bool)
    non_region = 1.0 - is_region.astype(np.float32)
    cnt = np.asarray(A @ non_region).ravel()
    return is_region & (cnt > 0)


def border_hops(A, border):
    """Unweighted shortest hop count from every node to the nearest border cell.

    Multi-source BFS via a virtual source node connected to all border cells.
    Unreachable cells return ``inf``.
    """
    from scipy.sparse import csr_matrix, hstack, vstack
    from scipy.sparse.csgraph import shortest_path

    border = np.asarray(border, dtype=bool)
    n = A.shape[0]
    if not border.any():
        return np.full(n, np.inf)
    idx = np.where(border)[0].astype(np.int32)
    top = csr_matrix((np.ones(idx.size, dtype=np.uint8),
                      (np.zeros(idx.size, dtype=np.int32), idx)),
                     shape=(1, n), dtype=np.uint8)
    aug = vstack([hstack([A, top.T.tocsr()]),
                  hstack([top, csr_matrix((1, 1), dtype=np.uint8)])]).tocsr()
    d = shortest_path(aug, method="D", directed=False, indices=n, unweighted=True)[:n]
    d = np.where(np.isinf(d), np.inf, np.maximum(d - 1.0, 0.0))
    d[border] = 0.0
    return d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def border_layers_from_coords(points, is_base, *, A=None, k=2, n_min=10,
                              ratio=0.2, max_edge=25.0):
    """Signed border-hop layer and signed micron distance for every cell.

    Parameters
    ----------
    points : (N, 2) array-like
        Cell centre coordinates (microns).
    is_base : (N,) bool array-like
        Whether each cell belongs to the base compartment (e.g. tumour).
    A : scipy.sparse matrix | None
        Precomputed symmetric 0/1 adjacency. When ``None`` a Delaunay graph is
        built from ``points`` (pruned at ``max_edge``).
    k, n_min, ratio : int, int, float
        Neighbourhood radius (hops), minimum neighbourhood size, and minimum
        base fraction used to call the base *region*.
    max_edge : float
        Delaunay edge-length cap (microns); ignored when ``A`` is supplied.

    Returns
    -------
    (signed_um, signed_hops) : tuple[np.ndarray, np.ndarray]
        ``signed_hops`` is negative inside the base region, 0 on the border,
        positive outside, ``nan`` where unreachable. ``signed_um`` is the exact
        distance to the nearest border-cell centroid with the same sign
        convention.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    is_base = np.asarray(is_base, dtype=bool)
    n = points.shape[0]

    if A is None:
        edges = delaunay_edges(points, max_edge=max_edge)
        A = adjacency_from_edges(n, edges)
    else:
        A = binarize_adjacency(A)

    Mk = _khop_reachability(A, k)
    region = base_region_mask(Mk, is_base, n_min, ratio)
    border = border_mask(A, region)
    hops = border_hops(A, border)

    signed_hops = hops.copy()
    signed_hops[region] *= -1.0
    signed_hops[~np.isfinite(hops)] = np.nan

    signed_um = np.full(n, np.nan, dtype=float)
    if border.any():
        tree = cKDTree(points[border])
        dist_um, _ = tree.query(points, k=1)
        signed_um = dist_um.astype(float)
        signed_um[region] *= -1.0
        signed_um[border] = 0.0
    signed_um[~np.isfinite(signed_hops)] = np.nan

    return signed_um, signed_hops
