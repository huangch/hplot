"""
h-plot: A spatial heterogeneity plot for tumor border analysis

This package provides:
- Core API for fitting and plotting spatial layer-based profiles
- Batch utilities for handling grouped plotting by tumor types
- CI computation adapted for varying region counts
"""

__version__ = "0.1.0"

from .core import HPlot
from .runners import run_hplot_batch
from .plotting import plot_hplot, plot_hplot_gam, plot_delta_hplot_gam
from .stats import (
    compute_layer_stats,
    compute_layer_pvalues,
    gam_group_curves,
    gam_pooled_effect,
    gam_delta_curve,
    binarize,
    cluster_mass_screen,
    compute_layer_kruskal_pvalues,
)

__all__ = [
    "HPlot",
    "run_hplot_batch",
    "plot_hplot",
    "plot_hplot_gam",
    "plot_delta_hplot_gam",
    "compute_layer_stats",
    "compute_layer_pvalues",
    "gam_group_curves",
    "gam_pooled_effect",
    "gam_delta_curve",
    "binarize",
    "cluster_mass_screen",
    "compute_layer_kruskal_pvalues",
    "pp",
    "tl",
    "pl",
    "io",
    "gr",
]

# scanpy-style AnnData API (pp/tl/pl) plus a CSV bridge (io). These submodules
# import anndata lazily *inside* their functions, so importing hplot (or
# hplot.core) still works without anndata installed. ``gr`` is a squidpy-style
# alias: hplot.gr.border_layers is hplot.pp.border_layers.
from . import pp, tl, pl, io, gr  # noqa: E402