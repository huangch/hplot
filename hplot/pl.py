"""Plotting: draw an H-Plot from ``adata.uns['hplot']`` or a saved CSV.

Mirrors ``scanpy.pl``: read the stored result and render it, returning the
matplotlib ``Axes`` so callers can further customise.

    hplot.pl.hplot(adata)                       # after hplot.tl.hplot
    hplot.pl.hplot_from_csv("hplot-outputs.csv")
"""

from __future__ import annotations

from ._serial import deserialize
from .io import read_hplot_csv
from .plotting import plot_hplot, plot_hpathway_summary


def hplot(adata, *, key="hplot", ax=None, ci_show=True, **kwargs):
    """Render the H-Plot stored by :func:`hplot.tl.hplot`.

    Extra keyword arguments override the stored plot settings and are forwarded
    to :func:`hplot.plotting.plot_hplot`.
    """
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns[{key!r}] missing; run hplot.tl.hplot(adata, ...) first."
        )
    stats, plot_kwargs = deserialize(adata.uns[key])
    plot_kwargs.update(kwargs)
    return plot_hplot(stats, ax=ax, ci_show=ci_show, **plot_kwargs)


def hplot_from_csv(path, *, group_col=None, ax=None, ci_show=True,
                   unit="um", value_kind="proportion",
                   display_base_type="tumour border",
                   display_target_type="target", **kwargs):
    """Read an ``hplot-outputs.csv`` and draw it in one call.

    Returns the matplotlib ``Axes``.
    """
    stats = read_hplot_csv(path, group_col=group_col)
    return plot_hplot(
        stats, ax=ax, ci_show=ci_show, unit=unit, value_kind=value_kind,
        display_base_type=display_base_type,
        display_target_type=display_target_type, **kwargs,
    )


def hpathway_summary_from_csv(path, *, fdr_col="fdr_dev", ax=None, **kwargs):
    """Read an H-Pathway Summary grid CSV and draw the H-Pathway Summary dotplot.

    The CSV is a tidy ``(pathway x layer)`` grid with ``pathway``, ``layer``,
    ``score`` and one or more FDR columns (``fdr_dev``, ``fdr_contrast``,
    ``fdr_treatment``, ``fdr_strata4``); ``fdr_col`` selects which one drives
    the dot alpha and rings. Extra keyword arguments are forwarded to
    :func:`hplot.plotting.plot_hpathway_summary`.

    Returns the dict from :func:`hplot.plotting.plot_hpathway_summary`.
    """
    import pandas as pd

    grid_df = pd.read_csv(path)
    return plot_hpathway_summary(grid_df, fdr_col=fdr_col, ax=ax, **kwargs)
