"""Graph namespace (squidpy-style alias).

hplot's border-layer assignment is a graph-geodesic operation over a spatial
neighbour graph, so squidpy users will reach for it under ``gr`` (next to
``sq.gr.spatial_neighbors``). This module re-exports the exact same function
objects as :mod:`hplot.pp`; ``hplot.gr.border_layers is hplot.pp.border_layers``.

    import squidpy as sq
    import hplot
    sq.gr.spatial_neighbors(adata)
    hplot.gr.border_layers(adata, "cell_type", ["tumour"])
    hplot.tl.hplot(adata, target="CD8A", groupby="cell_subtype")
    hplot.pl.hplot(adata)
"""

from __future__ import annotations

from .pp import border_layers

__all__ = ["border_layers"]
