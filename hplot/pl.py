"""Plotting: draw an H-Plot from ``adata.uns['hplot']`` or a saved CSV.

Mirrors ``scanpy.pl``: read the stored result and render it, returning the
matplotlib ``Axes`` so callers can further customise.

    hplot.pl.hplot(adata)                       # after hplot.tl.hplot
    hplot.pl.hplot_from_csv("hplot-outputs.csv")
"""

from __future__ import annotations

from ._serial import deserialize
from .io import read_hplot_csv
from .plotting import plot_hplot, plot_hpathway_dotplot


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


def hpathway_summary_from_csv(path, *, fdr_col=None, **kwargs):
    """Read a saved (pathway x layer) grid and draw the H-Pathway Summary.

    ``fdr_col`` defaults to ``None`` (position-only panel), which is the right
    reading when the pathway list was fixed elsewhere: pass the column name only
    when the grid carries a genuine competitive significance channel.
    """
    import pandas as pd

    return plot_hpathway_dotplot(pd.read_csv(path), fdr_col=fdr_col, **kwargs)


