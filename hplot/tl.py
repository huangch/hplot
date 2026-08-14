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


def _to_dense(block):
    """Densify a sparse row-block; pass dense arrays through unchanged."""
    todense = getattr(block, "todense", None)
    return todense() if todense is not None else block
