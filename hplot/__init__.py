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
from .plotting import (
    plot_hplot,
    plot_hplot_gam,
    plot_hplot_gam_delta,
    plot_hloci_strip,
    plot_hloci_bands,
    plot_hloci_fdr,
    plot_hloci_bands_bidir,
    plot_hloci_dotplot,
    plot_hpathway_dotplot,
    build_layer_distance_map,
    add_border_distance_axis,
)
from .stats import (
    compute_layer_stats,
    compute_layer_pvalues,
    gam_group_curves,
    gam_pooled_effect,
    gam_delta_curve,
    binarize,
    cluster_mass_screen,
    compute_layer_kruskal_pvalues,
    directional_cluster_bands,
    gradient_cluster_mass_screen,
    deviation_tensor,
    benjamini_hochberg,
    hpathway_summary_grid,
)
from .tl import ucell_scores, pathway_layer_profile, pathway_layer_profile_adata, pathway_layer_profile_h5ad
from ._geometry import border_layers_from_coords
from .catalogs import (
    load_catalog,
    read_gmt,
    write_gmt,
    select_signatures_on_panel,
)

__all__ = [
    "HPlot",
    "run_hplot_batch",
    "plot_hplot",
    "plot_hplot_gam",
    "plot_hplot_gam_delta",
    "plot_hloci_strip",
    "plot_hloci_bands",
    "plot_hloci_fdr",
    "plot_hloci_bands_bidir",
    "plot_hloci_dotplot",
    "plot_hpathway_dotplot",
    "build_layer_distance_map",
    "add_border_distance_axis",
    "compute_layer_stats",
    "compute_layer_pvalues",
    "gam_group_curves",
    "gam_pooled_effect",
    "gam_delta_curve",
    "binarize",
    "cluster_mass_screen",
    "compute_layer_kruskal_pvalues",
    "directional_cluster_bands",
    "gradient_cluster_mass_screen",
    "deviation_tensor",
    "benjamini_hochberg",
    "hpathway_summary_grid",
    "ucell_scores",
    "pathway_layer_profile",
    "pathway_layer_profile_adata",
    "pathway_layer_profile_h5ad",
    "border_layers_from_coords",
    "load_catalog",
    "read_gmt",
    "write_gmt",
    "select_signatures_on_panel",
    "catalogs",
    "pp",
    "tl",
    "pl",
    "io",
]

# scanpy-style AnnData API (pp/tl/pl) plus a CSV bridge (io). These submodules
# import anndata lazily *inside* their functions, so importing hplot (or
# hplot.core) still works without anndata installed.
from . import pp, tl, pl, io, catalogs  # noqa: E402