"""
h-plot: A spatial heterogeneity plot for tumor border analysis

This package provides:
- Core API for fitting and plotting spatial layer-based profiles
- Batch utilities for handling grouped plotting by tumor types
- CI computation adapted for varying region counts
"""

__version__ = "0.1.0"


def _harden_tqdm_against_resize() -> None:
    """Make every tqdm bar survive terminal / tmux resizes.

    1. ``dynamic_ncols=True`` becomes the default for every bar, so tqdm
       re-queries the terminal width on each refresh instead of caching the
       width it saw at construction time.
    2. A ``SIGWINCH`` handler redraws every live bar the moment the terminal
       is resized, rather than waiting for the next ``update()`` -- which on a
       permutation screen can be a long way off.

    Sentinels are deliberately NOT package-prefixed: these packages share one
    env and land in one process, so a per-package sentinel would let each of
    them wrap ``tqdm.__init__`` and chain another SIGWINCH handler. Keep this
    block identical across wsinsight, sptxinsight, hplot, kurtorank, wsitrain.
    """
    try:
        from tqdm import std as _tqdm_std
    except Exception:
        return

    if not getattr(_tqdm_std.tqdm, "_tqdm_resize_hardened", False):
        _orig_init = _tqdm_std.tqdm.__init__

        def _init(self, *args, **kwargs):  # noqa: ANN001
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("ascii", " =")
            _orig_init(self, *args, **kwargs)

        _tqdm_std.tqdm.__init__ = _init
        _tqdm_std.tqdm._tqdm_resize_hardened = True

    try:
        import os
        import signal

        if not hasattr(signal, "SIGWINCH"):
            return  # not POSIX (e.g. Windows); nothing to do
        if getattr(_tqdm_std.tqdm, "_tqdm_winch_installed", False):
            return

        _prev_handler = signal.getsignal(signal.SIGWINCH)

        def _on_winch(signum, frame):  # noqa: ANN001
            # tqdm falls back to COLUMNS/LINES when the ioctl fails (redirected
            # fp); a stale pair exported by the shell would pin the old width.
            os.environ.pop("COLUMNS", None)
            os.environ.pop("LINES", None)
            for inst in list(getattr(_tqdm_std.tqdm, "_instances", [])):
                # One bar that cannot be redrawn must not cost the others their
                # repaint, so each is isolated rather than the loop as a whole.
                try:
                    if inst.disable:
                        continue
                    pos = abs(inst.pos)
                    inst.moveto(pos)
                    # tqdm's own clear() blanks the line by writing as many
                    # spaces as the *old* width; once the terminal has shrunk
                    # that padding wraps and walks the bar down a row per
                    # resize. Erase to end of line, then drop the status
                    # printer so it stops padding to the pre-resize length.
                    inst.fp.write("\r\x1b[K")
                    inst.moveto(-pos)
                    inst.sp = inst.status_printer(inst.fp)
                    inst.refresh(nolock=True)
                except Exception:
                    continue
            if callable(_prev_handler):
                _prev_handler(signum, frame)

        signal.signal(signal.SIGWINCH, _on_winch)
        _tqdm_std.tqdm._tqdm_winch_installed = True
    except (ValueError, OSError):
        # signal.signal raises ValueError off the main thread; ignore.
        pass


_harden_tqdm_against_resize()

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
    pathway_competitive_test,
    hpathway_layer_ora,
    hpathway_score_grid,
    hpathway_arm_contrast,
)
from .pathways import (
    ucell_scores,
    pathway_layer_profile,
    pathway_layer_profile_h5ad,
)
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
    "pathway_competitive_test",
    "hpathway_layer_ora",
    "hpathway_score_grid",
    "hpathway_arm_contrast",
    "ucell_scores",
    "pathway_layer_profile",
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