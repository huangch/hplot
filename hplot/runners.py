import os
import numpy as np
import pandas as pd
from .core import HPlot

# Candidate column names for the per-layer count columns in an hplot-outputs.csv.
_TARGET_COUNT = ("target_count", "target_type_count")
_BASE_COUNT = ("base_count", "base_type_count")
_ALL_COUNT = ("all_count", "all_type_count", "n_cells")


def _resolve_col(cols, candidates, role):
    lut = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lut:
            return lut[cand]
    raise KeyError(
        f"exclude_base=True needs a {role} column; looked for {candidates} "
        f"but found {list(cols)}."
    )


def add_base_excluded_proportion(
    df,
    *,
    out_col="target_prop_base_excluded",
    target_count_col=None,
    base_count_col=None,
    all_count_col=None,
    min_base_excluded_count=1,
):
    """Derive a base-excluded proportion column from per-layer count columns.

    Computes ``target_count / (all_count - base_count)`` — the immune (target)
    fraction among **non-base** cells — instead of the default
    ``target_count / all_count``. Rows where ``all_count - base_count`` is below
    ``min_base_excluded_count`` are set to NaN (the engine drops them), so a
    layer that is almost entirely base cells cannot produce an unstable ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table carrying the three count columns (e.g. an hplot-outputs.csv).
    out_col : str
        Name of the derived proportion column to add.
    target_count_col, base_count_col, all_count_col : str | None
        Explicit count-column names; auto-detected when None.
    min_base_excluded_count : int
        Minimum non-base cell count required to keep a layer.

    Returns
    -------
    (pandas.DataFrame, str)
        A copy of ``df`` with ``out_col`` added, and the name of that column.
    """
    df = df.copy()
    tcol = target_count_col or _resolve_col(df.columns, _TARGET_COUNT, "target-count")
    bcol = base_count_col or _resolve_col(df.columns, _BASE_COUNT, "base-count")
    acol = all_count_col or _resolve_col(df.columns, _ALL_COUNT, "all-count")
    denom = df[acol].astype(float) - df[bcol].astype(float)
    valid = denom >= float(min_base_excluded_count)
    prop = pd.Series(np.nan, index=df.index, dtype=float)
    prop[valid] = df.loc[valid, tcol].astype(float) / denom[valid]
    df[out_col] = prop
    return df, out_col


def run_hplot_batch(
    df,
    targets="target_prop",
    layer="layer",
    group=None,
    distance=None,
    unit=None,
    ci=0.95,
    output="hplots",
    prefix="hplot",
    ci_show=True,
    format="svg",
    dpi=300,
    exclude_base=False,
    min_base_excluded_count=1,
):
    """
    根據群組自動產出多張 H-Plot 圖檔（每個 group 一張）

    When ``exclude_base=True`` the target proportion is recomputed from the
    ``target_count`` / ``base_count`` / ``all_count`` columns as
    ``target_count / (all_count - base_count)`` (base cells excluded from the
    denominator) and that derived column is plotted instead of ``targets``.
    """
    if exclude_base:
        df, targets = add_base_excluded_proportion(
            df, min_base_excluded_count=min_base_excluded_count)

    os.makedirs(output, exist_ok=True)

    if group and group in df.columns:
        groups = df[group].unique()
    else:
        groups = ["all"]

    for grp in groups:
        if grp == "all":
            sub_df = df
        else:
            sub_df = df[df[group] == grp]

        h = HPlot()
        h.fit(sub_df, targets=targets, layer=layer, group=group, distance=distance, unit=unit, ci=ci)
        h.plot(ci_show=ci_show)
        filename = os.path.join(output, f"{prefix}_{grp}.{format}")
        h.savefig(filename, dpi=dpi)