"""Serialise / deserialise a fitted HPlot into an h5ad-safe ``uns`` dict.

Everything is stored as fixed-name leaves (numpy arrays, strings, numbers) so
``adata.write_h5ad`` round-trips ``adata.uns['hplot']`` without a pickle **and**
without ever using a user label as a dict key. That last point matters: AnnData
writes nested-dict keys as HDF5 group names, and cell-type labels such as
``"T/NK cells"`` contain ``/`` (the HDF5 path separator), which would corrupt
the file. We therefore keep every per-group curve in flat columns tagged by an
integer ``group_index`` and map indices back to labels via ``group_order``.

:func:`deserialize` rebuilds the ``group -> stats DataFrame`` mapping that
:func:`hplot.plotting.plot_hplot` consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STAT_COLS = ("layer", "distance", "mean", "ci_lower", "ci_upper", "n")


def _color_to_hex(color):
    from matplotlib.colors import to_hex

    try:
        return to_hex(color, keep_alpha=False)
    except (ValueError, TypeError):
        return str(color)


def serialize(hp, *, value_kind, display_base_type, display_target_type,
              target, group_order=None, color_map=None, legend_title="Group"):
    """Turn a fitted :class:`hplot.core.HPlot` into a flat, h5ad-safe dict."""
    if group_order is not None:
        order = [str(k) for k in group_order]
        fitted = {str(k) for k in hp.target_grouped_stats_}
        order = [k for k in order if k in fitted]
        order += [k for k in (str(k) for k in hp.target_grouped_stats_)
                  if k not in set(order)]
    else:
        order = [str(k) for k in hp.target_grouped_stats_]

    # Flat columns tagged by group_index (no label ever becomes a dict key).
    col_arrays = {c: [] for c in _STAT_COLS}
    gidx = []
    for i, key in enumerate(order):
        sdf = hp.target_grouped_stats_[key]
        m = len(sdf)
        gidx.append(np.full(m, i, dtype=np.int64))
        for c in _STAT_COLS:
            if c in sdf.columns:
                col_arrays[c].append(np.asarray(sdf[c].to_numpy(), dtype=float))
            else:
                col_arrays[c].append(np.full(m, np.nan, dtype=float))

    def _cat(parts, dtype):
        return np.concatenate(parts) if parts else np.asarray([], dtype=dtype)

    stats = {"group_index": _cat(gidx, np.int64)}
    for c in _STAT_COLS:
        stats[c] = _cat(col_arrays[c], float)

    # Colours aligned to group_order (hex strings; "" when unspecified).
    cmap = {str(k): v for k, v in (color_map or {}).items()}
    colors = np.asarray(
        [_color_to_hex(cmap[k]) if k in cmap else "" for k in order],
        dtype=object,
    )

    out = {
        "stats": stats,
        "group_order": np.asarray(order, dtype=object),
        "colors": colors,
        "unit": "" if getattr(hp, "unit", None) is None else str(hp.unit),
        "value_kind": str(value_kind),
        "display_base_type": str(display_base_type),
        "display_target_type": str(display_target_type),
        "target": str(target),
        "legend_title": str(legend_title),
    }

    pv = getattr(hp, "layer_pvalues_", None)
    if pv is not None:
        out["pvalues"] = {c: np.asarray(pv[c].to_numpy(), dtype=float)
                          for c in pv.columns if pv[c].dtype.kind in "fiu"}
    return out


def deserialize(d):
    """Rebuild ``(target_grouped_stats, plot_kwargs)`` from a stored dict."""
    order = [str(k) for k in np.asarray(d["group_order"]).tolist()]
    st = d["stats"]
    gidx = np.asarray(st["group_index"])

    stats = {}
    for i, key in enumerate(order):
        mask = gidx == i
        stats[key] = pd.DataFrame(
            {c: np.asarray(st[c])[mask] for c in _STAT_COLS if c in st}
        ).reset_index(drop=True)

    colors = [str(c) for c in np.asarray(d.get("colors", [])).tolist()]
    color_map = {k: colors[i] for i, k in enumerate(order)
                 if i < len(colors) and colors[i]}
    color_map = color_map or None

    unit = str(d.get("unit", "")) or None
    plot_kwargs = dict(
        unit=unit,
        display_base_type=str(d.get("display_base_type", "tumor")),
        display_target_type=str(d.get("display_target_type", "target")),
        value_kind=str(d.get("value_kind", "proportion")),
        color_map=color_map,
        legend_order=order or None,
        legend_title=str(d.get("legend_title", "Group")),
    )
    return stats, plot_kwargs
