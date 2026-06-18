import pandas as pd
import numpy as np
from scipy.stats import t, norm, mannwhitneyu, ttest_ind

# Test name -> human-readable label used for the p-value axis.
PVALUE_TEST_LABELS = {
    "mannwhitney": "Mann-Whitney U",
    "ttest": "t-test",
    "welch": "Welch t-test",
}


def compute_layer_stats(df, prop, layer_col, distance_col, ci=0.95, use_t=True):
    grouped = df.groupby(layer_col)
    summary = []

    for layer, group in grouped:
        values = group[prop].values
        n = len(values)

        if n > 1:
            distance = group[distance_col].mean() if distance_col else None
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            sem = std / np.sqrt(n)

            if use_t or n <= 30:
                z = t.ppf(1 - (1 - ci) / 2, df=n - 1)
            else:
                z = norm.ppf(1 - (1 - ci) / 2)

            ci_lower = mean - z * sem
            ci_upper = mean + z * sem

            summary.append({
                'layer': layer,
                'distance': distance,
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n
            })
        else:
            distance = group[distance_col].mean() if distance_col else None
            mean = np.mean(values)
            
            summary.append({
                'layer': layer,
                'distance': distance,
                "mean": mean,
                "ci_lower": mean,
                "ci_upper": mean,
                "n": n   
            })        

    return pd.DataFrame(summary)


def _adjust_pvalues(pvals, method):
    """Multiple-testing correction over a 1-D array of p-values.

    NaNs are ignored (left as NaN). Supported methods: ``None`` (no change),
    ``"bonferroni"`` and ``"fdr_bh"`` (Benjamini-Hochberg). Implemented inline
    to avoid a statsmodels dependency.
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan, dtype=float)
    mask = ~np.isnan(pvals)
    p = pvals[mask]
    m = p.size
    if m == 0:
        return out

    if method is None:
        out[mask] = p
        return out
    if method == "bonferroni":
        out[mask] = np.minimum(p * m, 1.0)
        return out
    if method == "fdr_bh":
        order = np.argsort(p)
        ranked = p[order]
        adj = ranked * m / (np.arange(m) + 1)
        # enforce monotonicity from the largest p downwards
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
        result = np.empty(m, dtype=float)
        result[order] = adj
        out[mask] = result
        return out
    raise ValueError(
        f"Unknown correction '{method}'. Use None, 'bonferroni' or 'fdr_bh'."
    )


def compute_layer_pvalues(
    df,
    prop,
    layer_col,
    group_col,
    groups=None,
    test="mannwhitney",
    distance_col=None,
    min_n=3,
    correction=None,
):
    """Per-layer between-group p-value for a single target column.

    For each layer (unique value of ``layer_col``) the per-sample ``prop``
    values are split into the two groups and compared with a statistical test.
    The result is a tidy DataFrame with one row per layer, suitable for drawing
    a p-value track against a secondary log axis.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table with one row per (sample, layer).
    prop : str
        Column holding the per-sample target quantity to compare.
    layer_col : str
        Column with the (integer) layer index.
    group_col : str
        Column with the group label used to form the two arms.
    groups : tuple[Any, Any] | None
        The ordered pair of groups to compare. If ``None`` the two unique
        values of ``group_col`` are used (and exactly two are required).
    test : {"mannwhitney", "ttest", "welch"}
        Hypothesis test applied per layer. Default Mann-Whitney U (two-sided).
    distance_col : str | None
        Optional column with physical distance; its per-layer mean is carried
        through for secondary-axis tick labels.
    min_n : int
        Minimum number of non-NA samples required in *each* group for a layer
        to be tested. Layers below this still emit a row with ``p_value=NaN``.
    correction : {None, "bonferroni", "fdr_bh"}
        Optional multiple-testing correction across layers; result in ``p_adj``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``layer, distance, p_value, p_adj, stat, n1, n2`` sorted by
        layer.
    """
    for col in (prop, layer_col, group_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    if distance_col is not None and distance_col not in df.columns:
        raise KeyError(f"Column '{distance_col}' not found in DataFrame.")

    if groups is None:
        uniq = pd.unique(df[group_col].dropna())
        if len(uniq) != 2:
            raise ValueError(
                "compute_layer_pvalues needs exactly two groups; found "
                f"{len(uniq)} in '{group_col}'. Pass groups=(a, b) to choose a pair."
            )
        group_a, group_b = uniq[0], uniq[1]
    else:
        if len(groups) != 2:
            raise ValueError("groups must be a pair (group_a, group_b).")
        group_a, group_b = groups

    if test not in ("mannwhitney", "ttest", "welch"):
        raise ValueError(
            f"Unknown test '{test}'. Use 'mannwhitney', 'ttest' or 'welch'."
        )

    def _run_test(a, b):
        try:
            if test == "mannwhitney":
                res = mannwhitneyu(a, b, alternative="two-sided")
                return res.statistic, res.pvalue
            equal_var = test == "ttest"
            res = ttest_ind(a, b, equal_var=equal_var)
            return res.statistic, res.pvalue
        except ValueError:
            return np.nan, np.nan

    rows = []
    for layer, group in df.groupby(layer_col):
        distance = group[distance_col].mean() if distance_col else None
        a = group.loc[group[group_col] == group_a, prop].dropna().to_numpy()
        b = group.loc[group[group_col] == group_b, prop].dropna().to_numpy()
        n1, n2 = a.size, b.size

        if n1 >= min_n and n2 >= min_n:
            stat, pval = _run_test(a, b)
        else:
            stat, pval = np.nan, np.nan

        rows.append({
            "layer": layer,
            "distance": distance,
            "p_value": pval,
            "stat": stat,
            "n1": n1,
            "n2": n2,
        })

    out = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    out["p_adj"] = _adjust_pvalues(out["p_value"].to_numpy(), correction)
    return out