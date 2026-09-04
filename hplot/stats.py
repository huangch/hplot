import pandas as pd
import numpy as np
import warnings
from scipy.stats import (t, norm, mannwhitneyu, ttest_ind, chi2, rankdata,
                         kruskal, wilcoxon)

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

# ---------------------------------------------------------------------------
# Single-slide spatial-uniformity permutation test
# ---------------------------------------------------------------------------

def spatial_uniformity_test(
    cells_df,
    layer_col,
    target_col,
    n_perm=2000,
    seed=42,
    progress=True,
):
    """Single-slide spatial-uniformity permutation test.

    Tests whether a per-cell target quantity is spatially non-uniform across
    border layers within one slide. The statistic is the sum of squared
    deviations of the per-layer mean target from its grand mean::

        T = sum_L ( mean_L - mean_bar ) ** 2

    where ``mean_L`` is the mean of ``target_col`` among cells at layer L. The
    null distribution is built by shuffling the target values across cells
    (breaking the layer<->value association) ``n_perm`` times; the p-value is
    the fraction of permuted statistics >= the observed statistic.

    This is distinct from :func:`cluster_mass_screen`, which is a
    between-group Kruskal-Wallis cluster-mass test across patients.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Per-cell table with a layer column and a target column.
    layer_col : str
        Integer (or coercible) border-layer index column.
    target_col : str
        Per-cell target quantity (e.g. a 0/1 cell-type indicator).
    n_perm : int
        Number of shuffles for the null distribution. Default 2000.
    seed : int
        RNG seed. Default 42.
    progress : bool
        Show a tqdm progress bar if available. Default True.

    Returns
    -------
    dict
        ``observed_stat`` (float), ``perm_p`` (float),
        ``null_distribution`` (numpy.ndarray of length ``n_perm``).
    """
    rng_local = np.random.default_rng(seed)
    df = cells_df.dropna(subset=[layer_col]).copy()
    df[layer_col] = df[layer_col].astype(int)

    def _curve_stat(tgt_values):
        tmp = df[[layer_col]].copy()
        tmp["_tgt"] = np.asarray(tgt_values, dtype=float)
        props = tmp.groupby(layer_col)["_tgt"].mean()
        return float(np.sum((props - props.mean()) ** 2))

    tgt_vals = df[target_col].to_numpy(dtype=float).copy()
    observed_stat = _curve_stat(tgt_vals)

    iterator = range(n_perm)
    if progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(iterator, desc="perm", leave=False)
        except ImportError:
            pass

    null_dist = np.empty(n_perm)
    for i in iterator:
        rng_local.shuffle(tgt_vals)
        null_dist[i] = _curve_stat(tgt_vals)
    perm_p = float((null_dist >= observed_stat).mean())
    return dict(observed_stat=observed_stat, perm_p=perm_p,
                null_distribution=null_dist)


# ---------------------------------------------------------------------------
# Stage-2 GAM effect-size functions
# ---------------------------------------------------------------------------

from pygam import LinearGAM, s as _gam_s, l as _gam_l  # hard dependency

_DEFAULT_LAM_GRID = np.logspace(-3, 3, 11)


def gam_group_curves(
    long_df,
    target_col,
    layer_col,
    group_col,
    grid,
    groups=None,
    n_splines=10,
    lam_grid=None,
    ci_width=0.95,
):
    """Fit a penalised-spline GAM smooth curve per group (Stage-2 effect).

    For each group a separate model ``target ~ s(layer)`` is fitted using
    penalised B-splines.  The smoothing parameter lambda is chosen by GCV
    (generalised cross-validation) over ``lam_grid``.  The model is always
    fitted on the *full* layer range supplied in ``long_df``; you should not
    pre-filter to a Stage-1 cluster-mass band because that constitutes
    double-dipping and inflates the apparent effect.

    Mathematical background
    -----------------------
    B-spline basis: let **B**(l) be the (K x 1) vector of basis functions
    at layer l.  The smooth is f(l) = **B**(l)^T beta where beta is estimated
    by penalised least squares::

        beta* = argmin_beta  ||y - B beta||^2 + lambda * ||D^2 beta||^2

    D^2 is the second-difference matrix, penalising curvature.  lambda is
    chosen by GCV::

        lambda* = argmin_lambda  RSS(lambda) / [n * (1 - trace(H_lambda)/n)]^2

    where H_lambda is the hat matrix.  The 95 % pointwise CI is
    mean_pred +/- 1.96 * se_pred where se_pred is the square root of the
    diagonal of Var[f_hat] = B (B^T B + lambda D^T D)^{-1} B^T sigma^2.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Per-sample per-layer table (one row per sample x layer).
    target_col : str
        Response column (e.g. ``"immune_fraction"``).
    layer_col : str
        Integer layer index column.
    group_col : str
        Binary group label column.
    grid : array-like
        Layer values at which the fitted smooth is evaluated.
    groups : tuple | None
        Ordered ``(low, high)`` pair.  If ``None`` uses the two unique
        values of ``group_col`` in the order they appear.
    n_splines : int
        Number of B-spline basis functions (default 10).
    lam_grid : array-like | None
        Smoothing-penalty candidates for GCV. Default ``logspace(-3,3,11)``.
    ci_width : float
        Pointwise confidence-interval width (default 0.95).

    Returns
    -------
    dict[group_label, tuple[numpy.ndarray, numpy.ndarray]]
        ``{group: (pred, ci)}`` where *pred* is shape ``(len(grid),)`` and
        *ci* is shape ``(len(grid), 2)`` with columns ``[lower, upper]``.

    Raises
    ------
    ValueError
        If ``group_col`` does not contain exactly two groups (when *groups*
        is ``None``) or fewer than two samples are available for a group.
    """
    if lam_grid is None:
        lam_grid = _DEFAULT_LAM_GRID
    grid = np.asarray(grid, dtype=float)

    if groups is None:
        uniq = pd.unique(long_df[group_col].dropna())
        if len(uniq) != 2:
            raise ValueError(
                f"gam_group_curves needs exactly two groups; found {len(uniq)} "
                f"in '{group_col}'. Pass groups=(low, high)."
            )
        groups = (uniq[0], uniq[1])

    result = {}
    for grp in groups:
        sub = long_df[long_df[group_col] == grp].dropna(
            subset=[target_col, layer_col]
        )
        if len(sub) < 2:
            raise ValueError(
                f"Group '{grp}' has fewer than 2 rows after dropping NAs."
            )
        X = sub[layer_col].to_numpy(dtype=float)[:, None]
        y = sub[target_col].to_numpy(dtype=float)
        gam = LinearGAM(_gam_s(0, n_splines=n_splines)).gridsearch(
            X, y, lam=lam_grid, progress=False
        )
        Xg = grid[:, None]
        result[grp] = (
            gam.predict(Xg),
            gam.confidence_intervals(Xg, width=ci_width),
        )

    return result


def gam_pooled_effect(
    long_df,
    target_col,
    layer_col,
    group_col,
    at_layer,
    groups=None,
    covariate_cols=None,
    n_splines=10,
    lam_grid=None,
):
    """Pooled GAM: high-minus-low effect size at a given layer.

    Fits ``target ~ s(layer) + group [+ covariates]`` and returns the
    predicted difference between the high and low group at ``at_layer``
    together with the p-value of the linear group term.

    The group indicator and any covariates are entered as linear terms
    (``l(i)``) so their coefficients are interpretable marginal effects
    after accounting for the smooth non-linear layer trend.

    Mathematical background
    -----------------------
    The full model is::

        y_il = f(l) + beta_g * g_i + sum_k beta_k * x_{ik} + eps_il

    where:

    - f(l) = B(l)^T alpha  is a penalised B-spline smooth
    - g_i in {0, 1}  is the binary group indicator
    - x_{ik}  are optional linear covariates (z-scored internally)
    - eps ~ N(0, sigma^2)

    The design matrix is X = [B | g | x1 | x2 | ...].  All terms share a
    single GCV-chosen lambda (pygam gridsearch finds the optimal value).

    Effect size::

        Delta = f_hat(l0, g=1, x=x_bar) - f_hat(l0, g=0, x=x_bar)

    Covariates are evaluated at their mean (0 after z-scoring), so Delta
    is the group contrast at a typical patient at layer l0.

    The p-value is the Wald test for the linear group term (term index 1
    in pygam's term list), using the effective degrees of freedom from
    the GAM fit.

    Stage-1 double-dipping guard
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Pass the *full* layer range in ``long_df``.  Do **not** pre-filter to
    the Stage-1 cluster-mass band before calling this function; doing so
    selects on the outcome and inflates the apparent effect.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Per-sample per-layer table spanning the full layer range.
    target_col, layer_col, group_col : str
        Response, layer-index, and group-label columns.
    at_layer : int | float
        Layer at which the group effect is evaluated.  Use the Stage-1
        peak layer (e.g. the centroid of the cluster-mass band).
    groups : tuple | None
        ``(low_label, high_label)`` pair.  If ``None`` the two unique
        values of ``group_col`` are used in the order they appear.
    covariate_cols : list[str] | None
        Additional continuous / binary columns added as linear terms
        (e.g. ``["AGE", "late_stage", "is_female"]``).  Continuous
        covariates are z-scored internally for numerical stability.
    n_splines : int
        Number of B-spline basis functions (default 10).
    lam_grid : array-like | None
        Smoothing-penalty candidates. Default ``logspace(-3, 3, 11)``.

    Returns
    -------
    tuple[float, float, int]
        ``(effect, p_value, n)`` -- predicted high-minus-low response at
        ``at_layer``, GAM Wald p-value for the group term, number of rows
        used in the fit.

    Raises
    ------
    ValueError
        If required columns are missing or ``group_col`` does not contain
        exactly two groups.
    """
    if lam_grid is None:
        lam_grid = _DEFAULT_LAM_GRID
    covariate_cols = list(covariate_cols) if covariate_cols else []

    if groups is None:
        uniq = pd.unique(long_df[group_col].dropna())
        if len(uniq) != 2:
            raise ValueError(
                f"gam_pooled_effect needs exactly two groups; found {len(uniq)} "
                f"in '{group_col}'. Pass groups=(low, high)."
            )
        groups = (uniq[0], uniq[1])
    low_grp, high_grp = groups

    req_cols = [target_col, layer_col, group_col] + covariate_cols
    missing = [c for c in req_cols if c not in long_df.columns]
    if missing:
        raise ValueError(f"Columns not found in long_df: {missing}")

    df = long_df.dropna(subset=req_cols).copy()
    df["_grp01"] = df[group_col].map({low_grp: 0.0, high_grp: 1.0})
    df = df.dropna(subset=["_grp01"])

    # Z-score continuous covariates for numerical stability
    feature_cols = [layer_col, "_grp01"]
    for cov in covariate_cols:
        col_z = f"_z_{cov}"
        std = df[cov].std()
        df[col_z] = (df[cov] - df[cov].mean()) / std if std > 0 else 0.0
        feature_cols.append(col_z)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # s(layer) + l(group) [+ l(cov1) + l(cov2) + ...]
    terms = _gam_s(0, n_splines=n_splines) + _gam_l(1)
    for k in range(len(covariate_cols)):
        terms = terms + _gam_l(2 + k)

    gam = LinearGAM(terms).gridsearch(X, y, lam=lam_grid, progress=False)

    # Evaluate at at_layer; covariates at mean (== 0 after z-scoring)
    base = np.zeros((2, X.shape[1]))
    base[:, 0] = float(at_layer)
    base[0, 1] = 0.0   # low group
    base[1, 1] = 1.0   # high group

    effect = float(np.diff(gam.predict(base))[0])
    p_value = float(gam.statistics_["p_values"][1])   # term 1 = l(group)

    return effect, p_value, int(len(df))


def gam_delta_curve(curves, groups=None):
    """Pointwise difference curve Delta(layer) = pred_high - pred_low.

    Takes the output of :func:`gam_group_curves` and computes the layer-wise
    difference between the high and low group smooths, together with a
    propagated confidence interval and pointwise significance masks.

    Mathematical background
    -----------------------
    Given per-group GAM smooths f_lo(L) and f_hi(L) with pointwise 95 % CI
    half-widths sigma_lo(L) and sigma_hi(L) respectively, the difference is::

        Delta(L) = f_hi(L) - f_lo(L)

    The CI is propagated in quadrature (Gaussian error propagation, assumes
    independence between the two group models)::

        sigma_Delta(L) = sqrt(sigma_hi(L)^2 + sigma_lo(L)^2)

    so the CI bounds are Delta +/- sigma_Delta.

    .. warning::
        The significance masks ``sig_pos`` and ``sig_neg`` are based on
        **pointwise** (layer-by-layer) CIs only.  They are NOT corrected for
        multiple comparisons across layers.  Use them for visualisation and
        hypothesis generation, not for formal inference.

    Parameters
    ----------
    curves : dict
        Output of :func:`gam_group_curves`:
        ``{group_label: (pred_array, ci_array)}`` where *pred_array* has shape
        ``(n_grid,)`` and *ci_array* has shape ``(n_grid, 2)`` with columns
        ``[lower, upper]``.
    groups : tuple | None
        ``(low_label, high_label)`` identifying which key is the "low" group
        (subtracted) and which is the "high" group (added).  If ``None`` the
        first two keys in insertion order are used as ``(low, high)``.

    Returns
    -------
    diff_pred : numpy.ndarray, shape (n_grid,)
        Point estimate Delta(L) = pred_high(L) - pred_low(L).
    ci_lower : numpy.ndarray, shape (n_grid,)
        Lower bound of the propagated CI: Delta(L) - sigma_Delta(L).
    ci_upper : numpy.ndarray, shape (n_grid,)
        Upper bound of the propagated CI: Delta(L) + sigma_Delta(L).
    sig_pos : numpy.ndarray of bool, shape (n_grid,)
        ``True`` where ``ci_lower > 0``: high group is pointwise larger
        (CI excludes 0 from below).  NOT multiplicity-corrected.
    sig_neg : numpy.ndarray of bool, shape (n_grid,)
        ``True`` where ``ci_upper < 0``: low group is pointwise larger
        (CI excludes 0 from above).  NOT multiplicity-corrected.

    Raises
    ------
    ValueError
        If *curves* has fewer than two groups and *groups* is ``None``.
    KeyError
        If a label in *groups* is not a key of *curves*.

    Examples
    --------
    >>> grid = np.linspace(-7, 14, 200)
    >>> curves = gam_group_curves(long_df, "immune_frac", "layer", "hpv",
    ...                            grid, groups=("HPV-", "HPV+"))
    >>> diff, ci_lo, ci_hi, sig_pos, sig_neg = gam_delta_curve(
    ...     curves, groups=("HPV-", "HPV+"))
    """
    if groups is None:
        keys = list(curves.keys())
        if len(keys) < 2:
            raise ValueError(
                "gam_delta_curve: curves has fewer than two groups and "
                "groups=(low, high) was not supplied."
            )
        groups = (keys[0], keys[1])

    low_label, high_label = groups
    pred_lo, ci_lo = curves[low_label]
    pred_hi, ci_hi = curves[high_label]

    diff_pred = pred_hi - pred_lo

    # Quadrature CI propagation (Gaussian error propagation assuming independence)
    sigma_hi = (ci_hi[:, 1] - ci_hi[:, 0]) / 2.0
    sigma_lo = (ci_lo[:, 1] - ci_lo[:, 0]) / 2.0
    sigma_delta = np.sqrt(sigma_hi ** 2 + sigma_lo ** 2)

    ci_lower = diff_pred - sigma_delta
    ci_upper = diff_pred + sigma_delta

    sig_pos = ci_lower > 0   # high group dominates: CI excludes 0 from below
    sig_neg = ci_upper < 0   # low group dominates:  CI excludes 0 from above

    return diff_pred, ci_lower, ci_upper, sig_pos, sig_neg


# ── Cluster-mass spatial screen ──────────────────────────────────────────────

def binarize(series, min_per_group=10, method="median"):
    """Median split → 0 (low) / 1 (high) / -1 (unlabeled/missing).

    Parameters
    ----------
    series : array-like or pandas.Series
        Patient-level continuous values to split.
    min_per_group : int
        Minimum number of non-NaN values required in each group.  If the
        series has fewer than ``2 * min_per_group`` non-NaN values, all
        entries are returned as -1 (unlabeled).  Default 10.
    method : {"median"}
        Split method.  ``"median"`` assigns 0 to values strictly below the
        median and 1 to values strictly above.

    Returns
    -------
    numpy.ndarray of int
        Integer array of length ``len(series)`` with values in {-1, 0, 1}.
        -1 = NaN or data-insufficient; 0 = below median; 1 = above median.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    g = np.full(x.size, -1, dtype=int)
    ok = ~np.isnan(x)
    if ok.sum() < 2 * min_per_group:
        return g
    if method == "median":
        g[ok] = (x[ok] > np.nanmedian(x)).astype(int)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'median'.")
    return g


def _cms_best_band(H, thr, min_w):
    """Find the contiguous supra-threshold run with largest cluster mass."""
    L = len(H)
    supra = np.where(np.isnan(H), False, H > thr)
    best = 0.0; bs = be = -1; rm = 0.0; rl = 0; start = 0
    for l in range(L):
        if supra[l]:
            if rl == 0:
                start = l
            rm += H[l]; rl += 1
            if rl >= min_w and rm > best:
                best = rm; bs = start; be = l
        else:
            rm = 0.0; rl = 0
    if be < 0:
        return 0.0, (-1, -1, -1)
    peak = bs + int(np.nanargmax(H[bs:be + 1]))
    return best, (bs, be, peak)


def _cms_prep(mat, labeled, n_layers):
    """Pre-compute per-layer rank arrays for the cluster-mass screen."""
    M = mat.values
    idx_lab = np.where(labeled)[0]
    pos = -np.ones(M.shape[0], int)
    pos[idx_lab] = np.arange(idx_lab.size)
    cache = []
    for li in range(n_layers):
        col = M[:, li]
        present = labeled & ~np.isnan(col)
        gi = np.where(present)[0]
        if gi.size == 0:
            cache.append(None)
            continue
        vals = col[gi]
        R = rankdata(vals)
        N = vals.size
        _, cnt = np.unique(vals, return_counts=True)
        ties = cnt[cnt > 1]
        C = 1.0 - (ties ** 3 - ties).sum() / (N ** 3 - N) if N > 1 else 1.0
        cache.append((R, max(C, 1e-12), N, pos[gi], vals))
    return idx_lab, cache


def _cms_H_from(R, C, N, grp, k, min_per_group):
    """Kruskal-Wallis H statistic for one layer given pre-computed ranks."""
    s = 0.0
    for g in range(k):
        m = grp == g
        ng = int(m.sum())
        if ng < min_per_group:
            return np.nan
        s += R[m].sum() ** 2 / ng
    return (12.0 / (N * (N + 1)) * s - 3.0 * (N + 1)) / C


def cluster_mass_screen(
    mat,
    group_of,
    k,
    grid,
    cluster_alpha=0.05,
    min_cluster_w=1,
    min_per_group=10,
    n_perm=2000,
    seed=42,
    progress=True,
):
    """Kruskal-Wallis cluster-mass permutation test for spatial border profiles.

    For each column (layer) of *mat* a Kruskal-Wallis H-statistic is computed
    across the *k* patient groups defined by *group_of*.  Layers with H above
    the chi²(1−alpha, df=k-1) critical value are "supra-threshold"; the
    largest contiguous run of supra-threshold layers defines the *cluster mass*
    (sum of H values in the run).  Significance is calibrated by *n_perm*
    label permutations that preserve the spatial autocorrelation of the immune
    profile.

    Parameters
    ----------
    mat : pandas.DataFrame
        Wide patient × layer matrix. Rows = patients, columns = integer layer
        indices matching *grid*. NaN-tolerant (missing values are skipped per
        layer).
    group_of : array-like of int
        Group membership for each row of *mat*. Values in {0, …, k-1};
        -1 marks patients to exclude from the test.
    k : int
        Number of groups (must equal the number of distinct non-negative values
        in *group_of*).
    grid : array-like
        Layer grid corresponding to the columns of *mat* (e.g.
        ``np.arange(-7, 15)``).  Must satisfy ``len(grid) == mat.shape[1]``.
    cluster_alpha : float
        Per-layer cluster-forming significance threshold (chi² p-value).
        Default 0.05.
    min_cluster_w : int
        Minimum width (in layers) for a run to count as a cluster.  Default 1.
    min_per_group : int
        Minimum number of patients required in *each* group at a layer for
        that layer to receive a test statistic.  Default 10.
    n_perm : int
        Number of label-permutation draws for the null distribution.
        Default 2000.
    seed : int
        Seed for the permutation RNG.  Default 42.
    progress : bool
        Show a ``tqdm`` progress bar during permutations if the package is
        available.  Default True.

    Returns
    -------
    dict
        Keys:

        ``thr`` : float
            chi² critical value used as the cluster-forming threshold.
        ``H_obs`` : numpy.ndarray, shape (L,)
            Per-layer observed H-statistic (NaN where the layer was untested).
        ``grp_means`` : numpy.ndarray, shape (L, k)
            Per-layer per-group mean target proportion.
        ``mass`` : float
            Observed cluster mass (sum of H values in the best band).
        ``band`` : tuple (bs, be, peak)
            Start index, end index, and peak index into *grid* for the best
            cluster.  All -1 if no supra-threshold cluster was found.
        ``perm_p`` : float
            Permutation p-value, lower-bounded at ``1 / n_perm``.
        ``group_sizes`` : list of int
            Number of labeled patients in each group.
    """
    group_of = np.asarray(group_of, dtype=int)
    grid = np.asarray(grid)
    n_layers = len(grid)
    if mat.shape[1] != n_layers:
        raise ValueError(
            f"mat has {mat.shape[1]} columns but grid has {n_layers} elements."
        )

    labeled = group_of >= 0
    idx_lab, cache = _cms_prep(mat, labeled, n_layers)
    g_lab = group_of[idx_lab]
    thr = float(chi2.ppf(1.0 - cluster_alpha, df=k - 1))

    H_obs = np.full(n_layers, np.nan)
    grp_means = np.full((n_layers, k), np.nan)
    for li, c in enumerate(cache):
        if c is None:
            continue
        R, C, N, p, vals = c
        grp = g_lab[p]
        H_obs[li] = _cms_H_from(R, C, N, grp, k, min_per_group)
        for g in range(k):
            m = grp == g
            if m.sum():
                grp_means[li, g] = vals[m].mean()

    mass, (bs, be, pk) = _cms_best_band(H_obs, thr, min_cluster_w)

    # ── Permutation null distribution ──────────────────────────────────────
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    perm_iter = range(n_perm)
    if progress and _tqdm is not None:
        perm_iter = _tqdm(perm_iter, desc="label permutations", leave=False)

    for b in perm_iter:
        gp = rng.permutation(g_lab)
        Hn = np.full(n_layers, np.nan)
        for li, c in enumerate(cache):
            if c is None:
                continue
            R, C, N, p, _v = c
            Hn[li] = _cms_H_from(R, C, N, gp[p], k, min_per_group)
        null[b], _ = _cms_best_band(Hn, thr, min_cluster_w)

    perm_p = max(float((null >= mass).mean()), 1.0 / n_perm) if mass > 0 else 1.0

    return dict(
        thr=thr,
        H_obs=H_obs,
        grp_means=grp_means,
        mass=mass,
        band=(bs, be, pk),
        perm_p=perm_p,
        group_sizes=[int((g_lab == g).sum()) for g in range(k)],
    )


def compute_layer_kruskal_pvalues(
    df,
    prop,
    layer_col,
    group_col,
    groups=None,
    distance_col=None,
    min_n=3,
    correction="fdr_bh",
):
    """Per-layer Kruskal-Wallis p-values for k ≥ 2 groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table with one row per (sample, layer).
    prop : str
        Column holding the per-sample target quantity to compare.
    layer_col : str
        Column with the (integer) layer index.
    group_col : str
        Column with the group labels.
    groups : list | None
        Ordered list of group labels to compare. If ``None`` all unique
        non-NaN values in *group_col* are used (sorted).
    distance_col : str | None
        Optional column with physical distance; its per-layer mean is
        included in the output as ``distance``.
    min_n : int
        Minimum number of non-NaN samples required per group per layer for a
        test to be run.  Default 3.
    correction : {None, "bonferroni", "fdr_bh"}
        Multiple-testing correction applied across layers to produce the
        ``p_adj`` column.  Default ``"fdr_bh"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``layer, distance, p_value, p_adj, stat`` sorted by layer.
    """
    if groups is None:
        groups = sorted(df[group_col].dropna().unique().tolist())

    rows = []
    for layer, gdf in df.groupby(layer_col):
        arrs = [
            gdf.loc[gdf[group_col] == gname, prop].dropna().to_numpy()
            for gname in groups
        ]
        dist = gdf[distance_col].mean() if distance_col else None
        if all(a.size >= min_n for a in arrs) and len(arrs) >= 2:
            try:
                stat, p = kruskal(*arrs)
            except ValueError:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        rows.append({"layer": layer, "distance": dist, "p_value": p, "stat": stat})

    out = pd.DataFrame(rows).sort_values(layer_col).reset_index(drop=True)
    out["p_adj"] = _adjust_pvalues(out["p_value"].to_numpy(), correction)
    return out


# ---------------------------------------------------------------------------
# Single-group directional cluster-mass gradient screen (bidirectional H-loci)
# ---------------------------------------------------------------------------
#
# Terminology
# -----------
# Elevated H-domain
#     The maximum-mass *positive* contiguous supra-threshold deviation of a
#     feature relative to its analysis-window mean (``z_obs > 0``).
# Depressed H-domain
#     The maximum-mass *negative* contiguous supra-threshold deviation relative
#     to the analysis-window mean (``z_obs < 0``).
# Bidirectional H-locus representation
#     The pair of directional H-domains (elevated, depressed) reported for one
#     feature.  Either, both, or neither may exist / be significant.
# Dominant H-locus
#     The larger-mass of the two directional domains.
#
# Direction is always *relative to the feature's own mean within the predefined
# analysis window* (the layer range used to build the deviation tensor). An
# elevated and a depressed H-domain may simply be the two sides of a single
# monotonic spatial gradient rather than two independent biological programmes;
# a depressed band does **not** imply that every other layer is elevated, nor
# vice versa.


def _best_band_combined(h, supra, min_w):
    """Original winner-take-all band: largest-mass contiguous supra run.

    Direction-agnostic contiguous run over the (sign-agnostic) *supra* mask,
    maximising the summed unsigned mass ``h``.  This reproduces the historical
    ``_best_mass`` behaviour used by the ``band_mode="dominant"`` path.

    Returns ``(mass, start_idx, end_idx)`` with ``start_idx == end_idx == -1``
    when no qualifying run exists.
    """
    L = len(h)
    rm = 0.0
    rl = 0
    start = 0
    best = 0.0
    bs = be = -1
    for li in range(L):
        if supra[li]:
            if rl == 0:
                start = li
            rm += h[li]
            rl += 1
            if rl >= min_w and rm > best:
                best = rm
                bs = start
                be = li
        else:
            rm = 0.0
            rl = 0
    return best, bs, be


def _best_band_masked(h, mask, min_w):
    """Largest-mass contiguous run of ``h`` restricted to ``mask``.

    ``mask`` is a boolean directional supra-threshold mask (e.g.
    ``(z > 0) & (h > thr)``).  Runs are strictly contiguous: a single
    below-threshold / wrong-sign layer terminates the run.

    Returns ``(mass, start_idx, end_idx)`` with ``-1`` indices when empty.
    """
    return _best_band_combined(h, mask, min_w)


def _band_descriptor(h, z, bs, be, grid, direction):
    """Build a directional-band descriptor dict from band index bounds.

    ``center_of_mass`` is computed from the *unsigned* mass weights ``h`` over
    the band layers (not from the peak layer).  ``peak_idx`` / ``peak_layer``
    are retained separately for backward compatibility.
    """
    if bs < 0 or be < bs:
        return None
    idx = np.arange(bs, be + 1)
    hh = h[bs:be + 1]
    layers = np.asarray(grid, dtype=float)[bs:be + 1]
    mass = float(np.nansum(hh))
    peak_local = int(np.nanargmax(hh))
    peak_idx = bs + peak_local
    w = np.nan_to_num(hh, nan=0.0)
    wsum = float(w.sum())
    com_layer = float((layers * w).sum() / wsum) if wsum > 0 else float(np.nanmean(layers))
    return {
        "start_idx": int(bs),
        "end_idx": int(be),
        "start_layer": float(grid[bs]),
        "end_layer": float(grid[be]),
        "width_layers": int(be - bs + 1),
        "mass": mass,
        "peak_idx": int(peak_idx),
        "peak_layer": float(grid[peak_idx]),
        "center_of_mass": com_layer,
        "mean_signed_effect": float(np.nanmean(z[bs:be + 1])),
        "direction": direction,
    }


def directional_cluster_bands(z, thr=None, min_w=1, cluster_alpha=0.05, grid=None):
    """Detect the largest elevated and depressed H-domains of one profile.

    Treats positive and negative deviations as two *separate* directional
    searches over the same cluster-forming threshold, instead of squaring the
    signed statistic and keeping a single winner-take-all cluster.

    Parameters
    ----------
    z : array-like, shape (L,)
        Signed per-layer statistic (``z > 0`` elevated, ``z < 0`` depressed,
        relative to the analysis-window mean). NaNs are treated as sub-threshold.
    thr : float | None
        Cluster-forming threshold applied to the *unsigned* statistic
        ``h = z ** 2``.  When ``None`` it defaults to
        ``chi2.ppf(1 - cluster_alpha, df=1)`` — the same threshold used by the
        dominant-mode screen.
    min_w : int
        Minimum contiguous width (layers) for a run to count. Default 1.
    cluster_alpha : float
        Used only when ``thr is None``. Default 0.05.
    grid : array-like | None
        Layer coordinates aligned with ``z``. Defaults to ``arange(L)``.

    Returns
    -------
    dict
        ``{"elevated": desc | None, "depressed": desc | None, "thr": thr}``
        where each descriptor is the dict produced by :func:`_band_descriptor`.
    """
    z = np.asarray(z, dtype=float)
    L = z.size
    if grid is None:
        grid = np.arange(L)
    grid = np.asarray(grid, dtype=float)
    if grid.size != L:
        raise ValueError("grid must match the length of z.")
    if thr is None:
        thr = float(chi2.ppf(1.0 - cluster_alpha, df=1))
    h = np.nan_to_num(z ** 2, nan=0.0)
    supra = h > thr
    pos = supra & (z > 0)
    neg = supra & (z < 0)
    _, bs_p, be_p = _best_band_masked(h, pos, min_w)
    _, bs_n, be_n = _best_band_masked(h, neg, min_w)
    return {
        "elevated": _band_descriptor(h, z, bs_p, be_p, grid, "elevated"),
        "depressed": _band_descriptor(h, z, bs_n, be_n, grid, "depressed"),
        "thr": float(thr),
    }


def deviation_tensor(values, layers, grid, *, baseline_window=5,
                     min_baseline_layers=3, min_baseline_cells=50,
                     cell_counts=None, verbose=True):
    """Assemble the per-slide deviation tensor for the gradient screen.

    Each slide's per-layer values are placed onto a common ``grid`` (the
    cluster-mass analysis window) after subtracting a per-slide, per-unit
    **baseline**. The baseline *reference region* is selected with
    ``baseline_window`` and is independent of the layers that are actually
    tested for bands (those are always the ``grid`` layers):

    * ``"window"`` — baseline = mean over the slide's layers that fall inside
      ``grid``; deviations are relative to the slide's own window-wide level.
      Note that the tested layers then contribute to their own baseline, which
      bounds how far a band can rise above it while leaving the fall unbounded;
    * ``"far"`` — baseline = mean over *every* layer beyond the outer end of
      ``grid`` (``L > max(grid)``). Slides differ widely in how far their tissue
      extends, so this region is **not comparable across slides**; prefer an
      explicit width;
    * ``"core"`` — baseline = mean over every layer beyond the interior end of
      ``grid`` (``L < min(grid)``), i.e. deep inside the base object;
    * ``int`` — a fixed-width band anchored to the window edge, which is the
      recommended form because it gives every slide the same reference region
      without restating the edge. ``+k`` means the ``k`` layers immediately
      outside the window, ``(max(grid) + 1, max(grid) + k)``; ``-k`` means the
      ``k`` layers immediately inside it, ``(min(grid) - k, min(grid) - 1)``;
    * ``(a, b)`` — baseline = mean over the slide's layers with
      ``a <= L <= b`` (an explicit absolute reference window).

    There is no implicit default of ``None``: ``None`` is rejected because the
    baseline changes every number the screen reports. The default is ``5`` --
    the five layers immediately outside the window -- which is a stated choice,
    not an inherited one.

    A slide is dropped from the tensor (its whole row is NaN) when its baseline
    region is too thin to estimate a reference from, so an unstable reference
    never enters the pooled statistic. Two independent gates apply:
    ``min_baseline_layers`` counts *layers*, and ``min_baseline_cells`` counts
    the *cells* summed over those layers. They catch different failures: a
    section whose tissue ends just outside the window can still present the
    required number of layers while each is only a sliver a few cells wide.

    Parameters
    ----------
    values : sequence of ndarray, each ``(n_layers_slide, n_units)``
        Per-slide per-layer values (e.g. mean expression).
    layers : sequence of int ndarray, each ``(n_layers_slide,)``
        Integer layer coordinate for every row of the matching ``values``.
    grid : array-like, ``(n_layers,)``
        Layer coordinates of the analysis window (the tested layers).
    baseline_window : "window" | "far" | "core" | int | (int, int)
        Baseline reference-region selector (see above). Default ``5``.
    min_baseline_layers : int
        Minimum number of baseline-region *layers* a slide must contribute.
        Default 3 (a 3-layer average roughly halves the baseline noise vs a
        single layer). Raises if a fixed-width baseline is narrower than this,
        since no slide could then pass.
    min_baseline_cells : int
        Minimum number of *cells* summed across the slide's baseline-region
        layers. Default 50. Requires ``cell_counts``; ignored (with a warning
        when ``verbose``) if those are not supplied. Set to 0 to disable.
    cell_counts : sequence of ndarray | None
        Per-slide per-layer cell counts aligned with ``layers``, used by the
        ``min_baseline_cells`` gate. Default None.
    verbose : bool
        Print a one-line warning when one or more slides are skipped for an
        insufficient baseline region. Default True.

    Returns
    -------
    D : ndarray, shape ``(n_slides, n_layers, n_units)``
        Deviation tensor (NaN where a slide has no data / an unusable baseline).
    """
    grid = np.asarray(grid).astype(int)
    nG = grid.size
    gpos = {int(L): i for i, L in enumerate(grid)}
    lo, hi = int(grid.min()), int(grid.max())

    bw = baseline_window
    n_slides = len(values)
    if len(layers) != n_slides:
        raise ValueError("values and layers must have the same length.")

    n_units = None
    for V in values:
        V = np.asarray(V)
        if V.size:
            n_units = V.shape[1]
            break
    if n_units is None:
        raise ValueError("no slide contributes any values.")

    _BW_HELP = ("baseline_window must be 'window', 'far', 'core', an int "
                "(+k = the k layers just beyond max(grid); -k = the k layers "
                "just inside min(grid)), or an (a, b) layer range.")
    if bw is None:
        raise ValueError("baseline_window must be stated explicitly. " + _BW_HELP)
    if isinstance(bw, bool):
        raise ValueError(_BW_HELP)
    if isinstance(bw, (int, np.integer)):
        if int(bw) == 0:
            raise ValueError("baseline_window=0 has no width. " + _BW_HELP)
        k = int(bw)
        bw = (hi + 1, hi + k) if k > 0 else (lo + k, lo - 1)
    elif not (isinstance(bw, str) or (hasattr(bw, "__len__") and len(bw) == 2)):
        raise ValueError(_BW_HELP)

    if not isinstance(bw, str):
        _a, _b = int(bw[0]), int(bw[1])
        _width = _b - _a + 1
        if _width < min_baseline_layers:
            raise ValueError(
                f"baseline region [{_a}, {_b}] spans {_width} layer(s) but "
                f"min_baseline_layers={min_baseline_layers}, so no slide could "
                "ever pass. Widen the baseline or lower min_baseline_layers.")

    if cell_counts is not None and len(cell_counts) != n_slides:
        raise ValueError("cell_counts must have one entry per slide.")
    if min_baseline_cells and cell_counts is None and verbose:
        print(f"\u26a0 deviation_tensor: min_baseline_cells={min_baseline_cells} "
              "ignored because cell_counts was not supplied.")

    def _baseline_mask(lay):
        if bw == "window":
            return np.array([int(L) in gpos for L in lay], dtype=bool)
        if bw == "far":
            return lay > hi
        if bw == "core":
            return lay < lo
        try:
            a, b = bw
        except (TypeError, ValueError):
            raise ValueError(_BW_HELP)
        return (lay >= int(a)) & (lay <= int(b))

    D = np.full((n_slides, nG, n_units), np.nan, dtype=float)
    n_data = 0
    n_skip_layers = 0
    n_skip_cells = 0
    for si, (V, lay) in enumerate(zip(values, layers)):
        V = np.asarray(V, dtype=float)
        lay = np.asarray(lay).astype(int)
        if V.size == 0 or lay.size == 0:
            continue
        n_data += 1
        bmask = _baseline_mask(lay)
        if int(np.count_nonzero(bmask)) < min_baseline_layers:
            n_skip_layers += 1  # unstable / missing baseline -> skip this slide
            continue
        if cell_counts is not None and min_baseline_cells:
            nc = np.asarray(cell_counts[si], dtype=float)
            if nc.shape[0] != lay.shape[0]:
                raise ValueError(
                    f"cell_counts[{si}] has {nc.shape[0]} rows but layers has "
                    f"{lay.shape[0]}.")
            if float(np.nansum(nc[bmask])) < min_baseline_cells:
                n_skip_cells += 1  # layers present but only a sliver of tissue
                continue
        base = V[bmask].mean(axis=0)
        for k, L in enumerate(lay):
            gi = gpos.get(int(L))
            if gi is not None:
                D[si, gi] = V[k] - base
    if verbose and (n_skip_layers or n_skip_cells):
        print(f"\u26a0 deviation_tensor: skipped "
              f"{n_skip_layers + n_skip_cells}/{n_data} slide(s) in {bw!r} "
              f"baseline region ({n_skip_layers} with < {min_baseline_layers} "
              f"layer(s), {n_skip_cells} with < {min_baseline_cells} cell(s)); "
              "excluded from the pooled statistic.")
    return D


def _signed_layer_z(D, min_per_group=10):
    """Per-layer signed z across slides: ``mean / sem`` (NaN-tolerant).

    ``D`` is the ``(n_slides, n_layers, n_units)`` deviation tensor (each entry
    is a slide's per-layer value minus its analysis-window mean). A layer/unit
    receives a statistic only where at least ``min_per_group`` slides contribute
    and the standard error is positive; otherwise NaN.
    """
    n = np.sum(~np.isnan(D), axis=0)
    mu = np.nanmean(D, axis=0)
    sd = np.nanstd(D, axis=0, ddof=1)
    se = sd / np.sqrt(np.maximum(n, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where((n >= min_per_group) & (se > 0), mu / se, np.nan)
    return z


def _dominance_score(m_elev, m_depr):
    """Dominance / ambiguity measure for a gene's two directional masses.

    ``(m1 - m2) / m1`` with ``m1 = max`` and ``m2 = min`` when both directions
    exist; ``1.0`` when only one exists; ``NaN`` when neither exists. Close to 1
    means one direction clearly dominates; close to 0 means the two directions
    have similar mass and a single dominant-band summary would be ambiguous.
    """
    e = float(m_elev) if np.isfinite(m_elev) else 0.0
    d = float(m_depr) if np.isfinite(m_depr) else 0.0
    if e <= 0.0 and d <= 0.0:
        return np.nan
    if e <= 0.0 or d <= 0.0:
        return 1.0
    m1, m2 = (e, d) if e >= d else (d, e)
    return (m1 - m2) / m1


def _sign_flip_signs(n_slides, slide_groups, n_perm, rng):
    """Sign-flip design matrix whose exchangeable unit is a *group* of slides.

    Slides sharing a group (e.g. two runs from one patient) always flip together,
    so the null keeps their within-group correlation instead of breaking it. A
    per-slide flip would make the null less variable than the data whenever
    slides are clustered, which is anti-conservative.

    Returns ``(signs, n_groups, exact)`` where ``signs`` is ``(n_draw, n_slides)``.
    The whole sign space is enumerated when it is no larger than ``n_perm``, which
    turns the Monte-Carlo test into an exact one.
    """
    if slide_groups is None:
        gidx = np.arange(n_slides)
    else:
        slide_groups = np.asarray(slide_groups)
        if slide_groups.shape[0] != n_slides:
            raise ValueError("slide_groups must have one entry per slide.")
        _, gidx = np.unique(slide_groups, return_inverse=True)
    n_groups = int(gidx.max()) + 1
    if 2 ** n_groups <= n_perm:
        bits = (np.arange(2 ** n_groups)[:, None] >> np.arange(n_groups)) & 1
        return (bits * 2.0 - 1.0)[:, gidx], n_groups, True
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n_groups))
    return signs[:, gidx], n_groups, False


def gradient_cluster_mass_screen(
    D,
    grid,
    *,
    unit_names=None,
    band_mode="dominant",
    cluster_alpha=0.05,
    min_w=1,
    min_per_group=10,
    n_perm=1000,
    seed=0,
    layer_um=None,
    progress=False,
    slide_groups=None,
):
    """Single-group directional cluster-mass border-gradient screen.

    Generalises the notebook's pooled ``_screen_gradient`` into a reusable,
    testable routine.  For each unit (gene / interaction pair) it forms the
    signed per-layer statistic across slides, then either

    * ``band_mode="dominant"`` — reproduces the historical winner-take-all
      behaviour: one direction-agnostic maximum-mass band per unit, with a
      single pooled-permutation FDR; or
    * ``band_mode="bidirectional"`` — reports the largest **elevated** and the
      largest **depressed** band independently, each with its own directional
      permutation null and a BH-FDR taken across all unit x direction
      hypotheses.

    The permutation null is a slide-level **sign flip**: each slide's entire
    deviation block is multiplied by a random +-1, so every slide keeps its own
    layer autocorrelation, amplitude and NaN coverage while the cross-slide
    agreement that drives :func:`_signed_layer_z` is destroyed. Shuffling layers
    instead whitens the layer axis, which makes long contiguous supra-threshold
    runs much rarer under the null than in real (spatially smooth) profiles and
    leaves the test anti-conservative; it also relocates NaNs, so the
    ``min_per_group`` coverage gate would apply to different layers in the null
    than in the data. Signs are drawn once per slide and shared across units, so
    unit-unit covariance is preserved for the pooled-null FDR. The scheme is
    exact under symmetry of the slide-level deviation profile about zero, which
    :func:`deviation_tensor` enforces by centring each slide on its own baseline
    window.

    ``slide_groups`` sets the exchangeable unit. Leave it ``None`` when every
    slide is an independent biological replicate; pass a per-slide label (a
    patient identifier, say) whenever several slides come from the same subject,
    so that those slides flip together and the null retains their correlation.
    Flipping clustered slides independently makes the null less variable than the
    data and the test anti-conservative. The null space holds ``2 ** n_groups``
    states and is enumerated exhaustively -- giving an exact test -- whenever
    that is at most ``n_perm``.

    Parameters
    ----------
    D : numpy.ndarray, shape (n_slides, n_layers, n_units)
        Per-slide deviation tensor (value minus the slide's analysis-window
        mean). NaN-tolerant.
    grid : array-like, shape (n_layers,)
        Layer coordinates for the columns of ``D``.
    unit_names : sequence | None
        Names for the units (genes/pairs). Defaults to integer indices.
    band_mode : {"dominant", "bidirectional"}
        Selection behaviour (see above). Default ``"dominant"`` for backward
        compatibility.
    cluster_alpha : float
        Cluster-forming threshold ``chi2.ppf(1 - cluster_alpha, df=1)`` on the
        squared statistic. Default 0.05.
    min_w : int
        Minimum contiguous band width in layers. Default 1.
    min_per_group : int
        Minimum contributing slides per layer. Default 10.
    n_perm : int
        Number of sign-flip permutations. Default 1000.
    seed : int
        RNG seed. Default 0.
    layer_um : dict | None
        Optional ``{layer: micron}`` map to populate the ``*_um`` columns.
    progress : bool
        Show a tqdm bar over permutations if available. Default False.
    slide_groups : array-like | None
        Per-slide exchangeable-unit label. Default None (each slide its own).

    Returns
    -------
    dict
        ``thr`` : float
        ``z`` : ndarray (n_layers, n_units) signed per-layer statistic.
        ``long`` : DataFrame, one row per (unit, direction) that has a band.
        ``wide`` : DataFrame, one row per unit (elevated_* / depressed_* cols).
        ``band_mode`` : str echoing the mode used.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 3:
        raise ValueError("D must be a (n_slides, n_layers, n_units) tensor.")
    n_slides, n_layers, n_units = D.shape
    grid = np.asarray(grid)
    if grid.size != n_layers:
        raise ValueError("grid must match D's layer axis.")
    if band_mode not in ("dominant", "bidirectional"):
        raise ValueError("band_mode must be 'dominant' or 'bidirectional'.")
    if unit_names is None:
        unit_names = list(range(n_units))
    unit_names = list(unit_names)
    if len(unit_names) != n_units:
        raise ValueError("unit_names must match D's unit axis.")

    thr = float(chi2.ppf(1.0 - cluster_alpha, df=1))
    z_obs = _signed_layer_z(D, min_per_group)
    h_obs = np.nan_to_num(z_obs ** 2, nan=0.0)

    def _um(layer):
        if layer_um is None or not np.isfinite(layer):
            return np.nan
        return float(layer_um.get(int(round(layer)), np.nan))

    def _perm_iter(n):
        it = range(n)
        if progress:
            try:
                from tqdm.auto import tqdm as _tqdm
                return _tqdm(it, desc="perms", leave=False)
            except ImportError:
                return it
        return it

    rng = np.random.default_rng(seed)
    signs, n_groups, exact_null = _sign_flip_signs(n_slides, slide_groups, n_perm, rng)
    n_draw = signs.shape[0]
    meta = dict(n_groups=n_groups, n_draw=n_draw, exact_null=exact_null)

    if band_mode == "dominant":
        # ---- observed: combined winner-take-all band per unit ---------------
        obs_mass = np.zeros(n_units)
        bs_o = np.full(n_units, -1)
        be_o = np.full(n_units, -1)
        for u in range(n_units):
            m, bs, be = _best_band_combined(h_obs[:, u], h_obs[:, u] > thr, min_w)
            obs_mass[u], bs_o[u], be_o[u] = m, bs, be
        # ---- pooled sign-flip permutation null ------------------------------
        null = np.empty((n_draw, n_units), dtype=np.float32)
        for b in _perm_iter(n_draw):
            Dp = D * signs[b][:, None, None]
            hp = np.nan_to_num(_signed_layer_z(Dp, min_per_group) ** 2, nan=0.0)
            for u in range(n_units):
                null[b, u], _, _ = _best_band_combined(hp[:, u], hp[:, u] > thr, min_w)
        perm_p = (1.0 + (null >= obs_mass[None, :]).sum(0)) / (n_draw + 1.0)
        perm_p = np.where(obs_mass > 0, perm_p, 1.0)
        order = np.argsort(-obs_mass)
        obs_s = obs_mass[order]
        flat = np.sort(null.ravel())
        ge = flat.size - np.searchsorted(flat, obs_s, side="left")
        EV = ge / n_draw
        R = np.arange(1, n_units + 1)
        fdr_s = np.minimum(EV / R, 1.0)
        fdr_s = np.minimum.accumulate(fdr_s[::-1])[::-1]
        fdr = np.empty(n_units)
        fdr[order] = fdr_s
        fdr[obs_mass <= 0] = 1.0

        rows = []
        wide = []
        for u in range(n_units):
            desc = _band_descriptor(h_obs[:, u], z_obs[:, u], bs_o[u], be_o[u],
                                    grid, None)
            rec = {"gene": unit_names[u], "dominance_score": 1.0,
                   "permutation_p": float(perm_p[u]), "fdr": float(fdr[u])}
            if desc is None:
                wide.append({"gene": unit_names[u], "dominant_mass": 0.0,
                             "dominant_direction": "", "dominant_fdr": float(fdr[u]),
                             "dominance_score": np.nan})
                continue
            direction = "elevated" if z_obs[desc["peak_idx"], u] > 0 else "depressed"
            desc["direction"] = direction
            rows.append({**rec, "direction": direction,
                         "band_start_layer": desc["start_layer"],
                         "band_end_layer": desc["end_layer"],
                         "band_start_um": _um(desc["start_layer"]),
                         "band_end_um": _um(desc["end_layer"]),
                         "width_layers": desc["width_layers"],
                         "center_layer": desc["center_of_mass"],
                         "center_um": _um(round(desc["center_of_mass"])),
                         "peak_layer": desc["peak_layer"],
                         "cluster_mass": desc["mass"],
                         "mean_signed_effect": desc["mean_signed_effect"]})
            wide.append({"gene": unit_names[u], "dominant_mass": desc["mass"],
                         "dominant_direction": direction,
                         "dominant_center": desc["center_of_mass"],
                         "dominant_fdr": float(fdr[u]), "dominance_score": 1.0})
        long_df = pd.DataFrame(rows)
        wide_df = pd.DataFrame(wide)
        return dict(thr=thr, z=z_obs, long=long_df, wide=wide_df,
                    band_mode=band_mode, **meta)

    # ---- bidirectional --------------------------------------------------------
    obs_pos = np.zeros(n_units)
    obs_neg = np.zeros(n_units)
    desc_pos = [None] * n_units
    desc_neg = [None] * n_units
    for u in range(n_units):
        bands = directional_cluster_bands(z_obs[:, u], thr=thr, min_w=min_w, grid=grid)
        de, dd = bands["elevated"], bands["depressed"]
        desc_pos[u], desc_neg[u] = de, dd
        obs_pos[u] = de["mass"] if de else 0.0
        obs_neg[u] = dd["mass"] if dd else 0.0

    null_pos = np.zeros((n_draw, n_units), dtype=np.float32)
    null_neg = np.zeros((n_draw, n_units), dtype=np.float32)
    for b in _perm_iter(n_draw):
        Dp = D * signs[b][:, None, None]
        zp = _signed_layer_z(Dp, min_per_group)
        hp = np.nan_to_num(zp ** 2, nan=0.0)
        supra = hp > thr
        pos = supra & (zp > 0)
        neg = supra & (zp < 0)
        for u in range(n_units):
            null_pos[b, u], _, _ = _best_band_masked(hp[:, u], pos[:, u], min_w)
            null_neg[b, u], _, _ = _best_band_masked(hp[:, u], neg[:, u], min_w)

    # plus-one directional permutation p-values (never zero)
    p_pos = (1.0 + (null_pos >= obs_pos[None, :]).sum(0)) / (n_draw + 1.0)
    p_neg = (1.0 + (null_neg >= obs_neg[None, :]).sum(0)) / (n_draw + 1.0)
    p_pos = np.where(obs_pos > 0, p_pos, 1.0)
    p_neg = np.where(obs_neg > 0, p_neg, 1.0)

    # BH-FDR across all tested unit x direction hypotheses (mass > 0 only)
    tested = []
    pvals = []
    for u in range(n_units):
        if obs_pos[u] > 0:
            tested.append((u, "elevated"))
            pvals.append(p_pos[u])
        if obs_neg[u] > 0:
            tested.append((u, "depressed"))
            pvals.append(p_neg[u])
    fdr_map = {}
    if pvals:
        fdr_vals = _adjust_pvalues(np.asarray(pvals), "fdr_bh")
        for key, q in zip(tested, fdr_vals):
            fdr_map[key] = float(q)

    rows = []
    wide = []
    for u in range(n_units):
        dom = _dominance_score(obs_pos[u], obs_neg[u])
        wrec = {"gene": unit_names[u], "dominance_score": dom}
        for tag, desc, pv, obsm in (
            ("elevated", desc_pos[u], p_pos[u], obs_pos[u]),
            ("depressed", desc_neg[u], p_neg[u], obs_neg[u]),
        ):
            if desc is None or obsm <= 0:
                wrec[f"{tag}_start"] = np.nan
                wrec[f"{tag}_end"] = np.nan
                wrec[f"{tag}_center"] = np.nan
                wrec[f"{tag}_mass"] = 0.0
                wrec[f"{tag}_fdr"] = np.nan
                continue
            q = fdr_map.get((u, tag), np.nan)
            rows.append({
                "gene": unit_names[u], "direction": tag,
                "band_start_layer": desc["start_layer"],
                "band_end_layer": desc["end_layer"],
                "band_start_um": _um(desc["start_layer"]),
                "band_end_um": _um(desc["end_layer"]),
                "width_layers": desc["width_layers"],
                "width_um": (_um(desc["end_layer"]) - _um(desc["start_layer"])),
                "center_layer": desc["center_of_mass"],
                "center_um": _um(round(desc["center_of_mass"])),
                "peak_layer": desc["peak_layer"],
                "cluster_mass": desc["mass"],
                "mean_signed_effect": desc["mean_signed_effect"],
                "permutation_p": float(pv),
                "fdr": q,
                "dominance_score": dom,
                "analysis_window_start": float(grid[0]),
                "analysis_window_end": float(grid[-1]),
            })
            wrec[f"{tag}_start"] = desc["start_layer"]
            wrec[f"{tag}_end"] = desc["end_layer"]
            wrec[f"{tag}_center"] = desc["center_of_mass"]
            wrec[f"{tag}_mass"] = desc["mass"]
            wrec[f"{tag}_fdr"] = q
        wide.append(wrec)

    long_df = pd.DataFrame(rows)
    wide_df = pd.DataFrame(wide)
    return dict(thr=thr, z=z_obs, long=long_df, wide=wide_df,
                band_mode=band_mode, **meta)


# ---------------------------------------------------------------------------
# H-Pathway Summary: distance-stratified pathway/signature grid + FDR
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR for a 1-D array of p-values (NaN-safe).

    NaNs are ignored and returned as NaN; finite p-values are BH-adjusted with
    the usual monotone step-up and clipped to ``[0, 1]``. Thin public wrapper
    around the inline correction used throughout :mod:`hplot.stats` so callers
    do not need to re-implement it.

    Parameters
    ----------
    pvals : array-like
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        BH-adjusted q-values, same shape as ``pvals``.
    """
    return _adjust_pvalues(np.asarray(pvals, dtype=float), "fdr_bh")


def _pool_score_grid(profiles, path_names, grid, *, sample_col, layer_col,
                     score_agg="median"):
    """Pooled score per (layer, pathway): mean across slides of each slide's
    per-layer aggregate (equal weight per slide)."""
    gpos = {int(L): i for i, L in enumerate(grid)}
    nG, nP = len(grid), len(path_names)
    out = np.full((nG, nP), np.nan)
    for j, nm in enumerate(path_names):
        piv = profiles.pivot_table(index=sample_col, columns=layer_col,
                                   values=nm, aggfunc=score_agg)
        for L in grid:
            if L in piv.columns:
                col = piv[L].to_numpy(dtype=float)
                if np.isfinite(col).any():
                    out[gpos[int(L)], j] = np.nanmean(col)
    return out


def _deviation_fdr_grid(profiles, path_names, grid, *, sample_col, layer_col,
                        baseline_window=5, min_baseline_layers=3,
                        min_baseline_cells=50, count_col=None,
                        min_per_group=3, alternative="two-sided", verbose=True):
    """Per (layer, pathway) deviation FDR: per-slide deviation from a baseline
    region (``deviation_tensor``) tested per layer with a signed-rank Wilcoxon,
    BH-corrected over the whole grid.

    Returns ``(fdr, direction)``: ``fdr`` is the BH-corrected p-value grid and
    ``direction`` is the mean per-slide deviation from the baseline (positive =
    elevated vs baseline, negative = depressed), so callers can render a signed
    up/down glyph on the deviation panel too.
    """
    nG, nP = len(grid), len(path_names)
    vals, lays = [], []
    cnts = [] if count_col else None
    for _sid, sub in profiles.groupby(sample_col):
        sub = sub.sort_values(layer_col)
        vals.append(sub[path_names].to_numpy(dtype=float))
        lays.append(sub[layer_col].to_numpy(dtype=int))
        if cnts is not None:
            cnts.append(sub[count_col].to_numpy(dtype=float))
    Ddev = deviation_tensor(vals, lays, grid, baseline_window=baseline_window,
                            min_baseline_layers=min_baseline_layers,
                            min_baseline_cells=min_baseline_cells,
                            cell_counts=cnts, verbose=verbose)
    p_dev = np.full((nG, nP), np.nan)
    dir_dev = np.full((nG, nP), np.nan)
    for i in range(nG):
        for j in range(nP):
            x = Ddev[:, i, j]
            x = x[np.isfinite(x)]
            if x.size >= min_per_group and np.any(x != 0):
                try:
                    p_dev[i, j] = wilcoxon(x, zero_method="wilcox",
                                           alternative=alternative).pvalue
                except ValueError:
                    p_dev[i, j] = np.nan
                dir_dev[i, j] = float(np.mean(x))
    fdr = np.full(nG * nP, np.nan)
    flat = p_dev.ravel()
    mask = np.isfinite(flat)
    fdr[mask] = benjamini_hochberg(flat[mask])
    return fdr.reshape(nG, nP), dir_dev


def hpathway_summary_grid(profiles, *, path_names, grid,
                          sample_col="sample", layer_col="layer",
                          score_agg="median", deviation=5,
                          min_baseline_layers=3, min_baseline_cells=50,
                          count_col=None, min_per_group=3,
                          deviation_alternative="two-sided",
                          contrasts=None, long_df=None,
                          value_col="score", pathway_col="pathway",
                          coverage=None, verbose=True):
    """Build the tidy (pathway x layer) grid that feeds ``plot_hpathway_dotplot``.

    .. warning::

       ``fdr_dev`` is a **self-contained** test, not an enrichment test. It asks
       whether a set's score departs from *its own* baseline along the ruler --
       never whether it departs more than a size-matched random draw from the
       same measured universe would. On a targeted panel the second question is
       the one that carries information, because such panels are curated so that
       most of their genes track the contrast of interest; a self-contained test
       then calls a large fraction of sets significant regardless of what the
       sets contain, and the pathway *names* become uninterpretable. Use
       :func:`hpathway_layer_ora` (layer-resolved) or
       :func:`pathway_competitive_test` (pooled) before naming any row.

    Given per-slide, per-layer pathway/signature profiles this assembles the
    long grid the dotplot consumes, so a user does not re-implement the pooled
    score and the per-layer significance tests by hand:

    * ``score``     : pooled score = mean across slides of each slide's
      per-layer ``score_agg`` (equal weight per slide);
    * ``fdr_dev``   : deviation FDR = per-slide deviation from a baseline region
      (see :func:`deviation_tensor`) tested per layer with a signed-rank
      Wilcoxon, BH-corrected over the grid;
    * ``fdr_<name>``: one Kruskal-Wallis between-group contrast per entry in
      ``contrasts``, BH-corrected over the grid (with the raw ``p_<name>``).

    Parameters
    ----------
    profiles : pandas.DataFrame
        Wide per-(sample, layer) table with one column per pathway in
        ``path_names`` plus ``sample_col`` and ``layer_col``.
    path_names : sequence[str]
        Pathway/signature column names to place on the grid.
    grid : array-like[int]
        Layer coordinates of the analysis window (the tested layers).
    sample_col, layer_col : str
        Identifier columns in ``profiles``.
    score_agg : str
        Per-slide per-layer aggregate for the pooled score. Default ``"median"``.
    deviation : "window" | "far" | "core" | int | (int, int) | "skip"
        Baseline reference-region selector for the deviation FDR
        (see :func:`deviation_tensor`). Default ``5``. ``"skip"`` omits
        ``fdr_dev`` altogether. ``None`` raises.
    min_baseline_layers, min_per_group : int
        Deviation baseline layer floor and minimum non-NaN samples per test.
    min_baseline_cells : int
        Minimum cells summed over a slide's baseline layers. Needs ``count_col``.
    count_col : str | None
        Column of ``profiles`` holding per-(sample, layer) cell counts, used by
        the ``min_baseline_cells`` gate. Default None (gate inactive).
    deviation_alternative : str
        Wilcoxon alternative for the deviation test.
    contrasts : dict[str, tuple[str, sequence]] | None
        ``{name: (group_col, groups)}``; each adds ``p_<name>`` and
        ``fdr_<name>`` from a per-layer Kruskal-Wallis test across ``groups``.
    long_df : pandas.DataFrame | None
        Tidy long table (``pathway_col``, ``layer_col``, ``value_col`` and the
        contrast group columns) used for the contrasts. If ``None`` it is
        melted from ``profiles`` (id columns = every non-pathway column).
    value_col, pathway_col : str
        Value / pathway column names in ``long_df``.
    coverage : dict[str, tuple[int, int]] | None
        Optional ``{pathway: (n_measured, n_total)}``. When given, a low median
        coverage raises a warning: a set represented by a handful of probes is
        named after a program it cannot measure.

    Returns
    -------
    pandas.DataFrame
        Long grid with columns ``pathway, layer, score`` (+ ``fdr_dev`` and
        ``p_<name>``/``fdr_<name>`` per contrast), one row per (pathway, layer).
    """
    path_names = list(path_names)
    grid = [int(L) for L in grid]
    nG, nP = len(grid), len(path_names)

    if coverage:
        import warnings
        fr = [m / t for m, t in (coverage.get(nm, (0, 0)) for nm in path_names) if t]
        if fr and float(np.median(fr)) < 0.20:
            warnings.warn(
                f"median gene-set coverage is {100 * np.median(fr):.0f}% of set size; "
                "set scores are proxies for the few measured members, so pathway "
                "names carry little information. Use pathway_competitive_test to "
                "decide which rows may be named.", stacklevel=2)

    score_grid = _pool_score_grid(profiles, path_names, grid,
                                   sample_col=sample_col, layer_col=layer_col,
                                   score_agg=score_agg)

    if deviation is None:
        raise ValueError(
            'deviation must be stated explicitly: "window", "far", "core", an '
            'int width, an (a, b) range, or "skip" to omit the deviation '
            'channel. None is rejected because it reads as a baseline choice '
            'but would silently drop the fdr_dev column.')
    if isinstance(deviation, str) and deviation == "skip":
        fdr_dev = dir_dev = None
    else:
        fdr_dev, dir_dev = _deviation_fdr_grid(
            profiles, path_names, grid, sample_col=sample_col,
            layer_col=layer_col, baseline_window=deviation,
            min_baseline_layers=min_baseline_layers,
            min_baseline_cells=min_baseline_cells, count_col=count_col,
            min_per_group=min_per_group,
            alternative=deviation_alternative, verbose=verbose)

    rows = []
    for i, L in enumerate(grid):
        for j, nm in enumerate(path_names):
            rec = dict(pathway=nm, layer=int(L), score=score_grid[i, j])
            if fdr_dev is not None:
                rec["fdr_dev"] = fdr_dev[i, j]
                rec["dir_dev"] = dir_dev[i, j]
            rows.append(rec)
    grid_df = pd.DataFrame(rows)

    if not contrasts:
        return grid_df

    if long_df is None:
        id_cols = [c for c in profiles.columns if c not in path_names]
        long_df = profiles.melt(id_vars=id_cols, value_vars=path_names,
                                var_name=pathway_col, value_name=value_col)

    for cname, (group_col, groups) in contrasts.items():
        parts = []
        dir_parts = []
        groups = tuple(groups)
        for nm in path_names:
            sub = long_df[long_df[pathway_col] == nm]
            try:
                kp = compute_layer_kruskal_pvalues(
                    sub, value_col, layer_col, group_col,
                    groups=tuple(groups), min_n=min_per_group, correction=None)
            except Exception:
                continue
            kp = kp[[layer_col, "p_value"]].copy()
            kp[pathway_col] = nm
            parts.append(kp)
            # Signed direction for two-group contrasts: mean(groups[1]) minus
            # mean(groups[0]) per layer (equal weight per sample), so a plot can
            # show which side is higher. Undefined for >2 groups -> skipped.
            if len(groups) == 2:
                gm = (sub[sub[group_col].isin(groups)]
                      .groupby([layer_col, group_col])[value_col].mean()
                      .unstack(group_col))
                if set(groups).issubset(gm.columns):
                    dd = (gm[groups[1]] - gm[groups[0]]).rename("dir").reset_index()
                    dd[pathway_col] = nm
                    dir_parts.append(dd)
        pcol, fcol, dcol = f"p_{cname}", f"fdr_{cname}", f"dir_{cname}"
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.rename(columns={"p_value": pcol, layer_col: "layer",
                                            pathway_col: "pathway"})
            grid_df = grid_df.merge(merged[["layer", "pathway", pcol]],
                                    on=["layer", "pathway"], how="left")
            m = grid_df[pcol].notna().to_numpy()
            grid_df[fcol] = np.nan
            if m.any():
                grid_df.loc[m, fcol] = benjamini_hochberg(
                    grid_df.loc[m, pcol].to_numpy())
        else:
            grid_df[pcol] = np.nan
            grid_df[fcol] = np.nan
        if dir_parts:
            dmerged = pd.concat(dir_parts, ignore_index=True)
            dmerged = dmerged.rename(columns={layer_col: "layer",
                                              pathway_col: "pathway"})
            grid_df = grid_df.merge(dmerged[["layer", "pathway", "dir"]],
                                    on=["layer", "pathway"], how="left")
            grid_df = grid_df.rename(columns={"dir": dcol})

    return grid_df


def pathway_competitive_test(gene_stat, gene_sets, *, hits=None, n_draw=10000,
                             seed=0, min_genes=3):
    """Competitive gene-set test against size-matched draws from the same universe.

    The companion to :func:`hpathway_summary_grid`, whose ``fdr_dev`` channel is
    self-contained. Here each set is scored against random sets of the same size
    drawn from the genes that were actually measured, which is the question an
    enrichment claim needs answered.

    The universe is ``gene_stat``'s keys and must stay that way. Substituting a
    transcriptome-scale background for a targeted panel is not a fix but a much
    larger bias: the untested genes enter the denominator as non-significant and
    almost every set becomes spuriously enriched.

    Parameters
    ----------
    gene_stat : Mapping[str, float]
        Per-gene statistic for every measured gene (e.g. cluster mass from
        :func:`gradient_cluster_mass_screen`). Its keys define the universe.
    gene_sets : Mapping[str, Sequence[str]]
        Candidate sets. Genes outside the universe are dropped, and sets left
        with fewer than ``min_genes`` members are skipped.
    hits : Iterable[str] | None
        Genes individually called significant. When given, an over-representation
        test is added alongside the statistic-based one.
    n_draw : int
        Size-matched random draws per set. Default 10000.
    seed : int
        RNG seed.
    min_genes : int
        Minimum measured members for a set to be testable. Default 3.

    Returns
    -------
    pandas.DataFrame
        One row per testable set: ``n_measured``, ``mean_stat``,
        ``null_mean_stat``, ``p_stat``, ``q_stat`` and -- when ``hits`` is given
        -- ``n_hits``, ``expected_hits``, ``p_overrep``, ``q_overrep``.
    """
    from scipy.stats import hypergeom

    universe = list(gene_stat)
    pos = {g: i for i, g in enumerate(universe)}
    stat = np.asarray([float(gene_stat[g]) for g in universe])
    hit_vec = (np.asarray([g in set(hits) for g in universe])
               if hits is not None else None)
    rng = np.random.default_rng(seed)

    rows = []
    for name, genes in gene_sets.items():
        idx = np.asarray([pos[g] for g in dict.fromkeys(genes) if g in pos], dtype=int)
        if idx.size < min_genes:
            continue
        k = int(idx.size)
        obs = float(stat[idx].mean())
        null = stat[rng.choice(len(universe), size=(n_draw, k), replace=True)].mean(1)
        rec = dict(pathway=name, n_measured=k, mean_stat=obs,
                   null_mean_stat=float(null.mean()),
                   p_stat=(1 + int((null >= obs).sum())) / (n_draw + 1))
        if hit_vec is not None:
            n_hit = int(hit_vec[idx].sum())
            rec.update(n_hits=n_hit,
                       expected_hits=float(k * hit_vec.mean()),
                       p_overrep=float(hypergeom.sf(n_hit - 1, len(universe),
                                                    int(hit_vec.sum()), k)))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("stat", "overrep"):
        if f"p_{col}" in out:
            out[f"q_{col}"] = _adjust_pvalues(out[f"p_{col}"].to_numpy(float), "fdr_bh")
    return out.sort_values("p_stat", kind="mergesort").reset_index(drop=True)


def hpathway_layer_ora(gene_bands, gene_sets, *, grid, gene_col="gene",
                       fdr_col="fdr_global", band_lo_col="band_lo",
                       band_hi_col="band_hi", alpha=0.05, min_genes=5,
                       min_run=2, verbose=True):
    """Per-layer over-representation of a gene set among border-band genes.

    This is the layer-resolved form of :func:`pathway_competitive_test`, and the
    channel to report instead of ``hpathway_summary_grid``'s ``fdr_dev``.

    Why aggregate counts rather than profiles
    -----------------------------------------
    Averaging a set's member *profiles* and testing the average is not usable
    here. Tissue-level gradients act on every gene, so any set of genes yields a
    smooth, reproducible, non-flat profile; worse, the members of a real set are
    co-expressed, so their average cancels less noise than a random set's does
    and the set is compared against a null that is too tight. Measured on a
    5046-gene bladder panel the variance inflation factor of the 50 Hallmark
    sets ran from 3.0 to 77 (median 18), i.e. a set of 73 genes behaved like
    ~4 independent ones.

    Counting sidesteps both problems. Each gene has already been tested on its
    own against the screen's permutation null, so the shared gradient is removed
    *before* any set-level aggregation, and the set statistic is then a
    hypergeometric count rather than an average of correlated curves.

    The per-layer background is not flat and that is the point
    ---------------------------------------------------------
    The fraction of the panel carrying a band varies strongly along the ruler --
    on the bladder cohort, 7-8% of genes at layers -5..+1 against 42-48% at
    +3..+15. A pooled test compares every set against one global rate and cannot
    see this; testing layer by layer compares each set against the rate that
    actually applies there.

    Parameters
    ----------
    gene_bands : pandas.DataFrame
        One row per **measured** gene, as produced by
        :func:`gradient_cluster_mass_screen`: a gene id, an FDR, and the
        inclusive band limits. Its rows define the universe, which must stay the
        measured panel -- substituting a transcriptome-wide background lets the
        unmeasured genes enter the denominator as non-significant and makes
        almost every set look enriched.
    gene_sets : Mapping[str, Sequence[str]]
        Candidate sets. Members outside the universe are dropped.
    grid : array-like[int]
        Layers to test.
    alpha : float
        FDR below which a gene counts as carrying a band. Default 0.05.
    min_genes : int
        Minimum measured members for a set to be tested.
    min_run : int
        Consecutive significant layers required to call a set significant. A
        single isolated layer is not a band, and the same contiguity requirement
        is what the gene-level cluster-mass screen applies. This rule does real
        work: on the bladder cohort ``Spermatogenesis`` reaches q = 3.2e-3 at one
        layer and is rejected here for having no second one.

    Returns
    -------
    grid_df : pandas.DataFrame
        One row per (pathway, layer): ``k``, ``hits``, ``expected``,
        ``background_frac``, ``p``, ``q`` (BH over the whole grid).
    summary : pandas.DataFrame
        One row per pathway: ``n_measured``, ``n_sig_layers``, ``max_run``,
        ``best_layer``, ``best_q``, ``significant``.

    Notes
    -----
    Two approximations are inherited from over-representation analysis and
    should be stated wherever results are reported. The hypergeometric assumes
    genes are independent, so co-expression makes the counts over-dispersed and
    p-values mildly anti-conservative; and a gene whose band spans several
    layers is counted at each of them, so a pathway's layer tests are strongly
    dependent (BH remains valid under positive dependence).
    """
    from scipy.stats import hypergeom

    grid = np.asarray(grid).astype(int)
    need = {gene_col, fdr_col, band_lo_col, band_hi_col}
    missing = need - set(gene_bands.columns)
    if missing:
        raise ValueError(f"gene_bands is missing column(s): {sorted(missing)}")

    genes = gene_bands[gene_col].astype(str).to_numpy()
    pos = {g: i for i, g in enumerate(genes)}
    N = genes.size
    lo = gene_bands[band_lo_col].to_numpy(dtype=float)
    hi = gene_bands[band_hi_col].to_numpy(dtype=float)
    sig = (gene_bands[fdr_col].to_numpy(dtype=float) < alpha) & np.isfinite(lo)
    cover = np.array([sig & (lo <= L) & (hi >= L) for L in grid])   # (n_grid, N)
    K = cover.sum(axis=1)

    rows, kept = [], []
    for name, members in gene_sets.items():
        idx = np.array([pos[g] for g in dict.fromkeys(members) if g in pos],
                       dtype=int)
        if idx.size < min_genes:
            continue
        k = int(idx.size)
        kept.append((name, k))
        for i, L in enumerate(grid):
            x = int(cover[i][idx].sum())
            rows.append(dict(pathway=name, layer=int(L), k=k, hits=x,
                             expected=float(k * K[i] / N),
                             background_frac=float(K[i] / N),
                             p=float(hypergeom.sf(x - 1, N, int(K[i]), k))))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    grid_df = pd.DataFrame(rows)
    grid_df["q"] = _adjust_pvalues(grid_df["p"].to_numpy(float), "fdr_bh")

    out = []
    for name, k in kept:
        sub = grid_df[grid_df.pathway == name].sort_values("layer")
        hit = (sub["q"].to_numpy() < alpha)
        run = best = 0
        for h in hit:
            run = run + 1 if h else 0
            best = max(best, run)
        j = int(np.argmin(sub["q"].to_numpy()))
        out.append(dict(pathway=name, n_measured=k, n_sig_layers=int(hit.sum()),
                        max_run=int(best),
                        best_layer=int(sub["layer"].to_numpy()[j]),
                        best_q=float(sub["q"].to_numpy()[j]),
                        significant=bool(best >= min_run)))
    summary = pd.DataFrame(out).sort_values(
        ["significant", "max_run", "best_q"], ascending=[False, False, True],
        kind="mergesort").reset_index(drop=True)

    if verbose:
        print(f"per-layer ORA: {int(summary['significant'].sum())}/"
              f"{len(summary)} sets with >= {min_run} consecutive layers at "
              f"q<{alpha} | {int((grid_df['q'] < alpha).sum())} of "
              f"{len(grid_df)} cells | universe {N} genes, "
              f"{int(sig.sum())} with a band ({100 * sig.mean():.1f}%)")
    return grid_df, summary


def hpathway_score_grid(profiles, *, path_names, grid, sample_col="sample",
                        layer_col="layer", score_agg="median", baseline="window",
                        min_baseline_layers=3, min_baseline_cells=50,
                        count_col=None, verbose=True):
    """Tidy (pathway x layer) activity grid for an already-chosen signature list.

    The companion to :func:`hpathway_layer_ora`, for the case where *which*
    pathways matter has been settled elsewhere -- typically on a different assay
    with a neutral gene background -- and the only remaining question is **where
    each one sits on the border ruler**. Nothing here selects, ranks or tests a
    pathway, so the output cannot be read as enrichment.

    Two channels are returned, both directionless:

    * ``score``     : pooled activity = mean across units of each unit's
      per-layer ``score_agg`` (equal weight per unit);
    * ``deviation`` : the same pooled value after each unit is centred on its own
      baseline region (:func:`deviation_tensor`), which is what makes profiles
      from units of different overall level comparable.

    ``deviation`` is signed, but the sign is **not** a pathway direction: a gene
    set contains positively and negatively regulated members, so the mean of its
    members is not a statement that the program is elevated or depressed. Centring
    also forces the sign to flip somewhere along the ruler when ``baseline`` is
    ``"window"`` (each unit sums to ~0 over the tested layers), so only the
    *position* of the crossing carries information -- never its existence, and
    never its direction. Plot with ``direction_col=None``.

    No significance channel is produced. The earlier ``fdr_dev`` test asked
    whether a set departed from *its own* baseline, which on a targeted panel is
    close to preordained and made pathway names uninterpretable; when a p-value
    is needed, compare a set against size-matched random draws from the same
    panel (:func:`pathway_competitive_test`) or count band-carrying members per
    layer (:func:`hpathway_layer_ora`).

    Parameters
    ----------
    profiles : pandas.DataFrame
        Wide per-(unit, layer) table with one column per name in ``path_names``
        plus ``sample_col`` and ``layer_col``. The unit should be the level at
        which observations are independent (usually the patient, not the slide).
    path_names : sequence[str]
        Signature columns to place on the grid.
    grid : array-like[int]
        Layer coordinates of the analysis window.
    score_agg : str
        Per-unit per-layer aggregate for the pooled score. Default ``"median"``.
    baseline : "window" | "far" | "core" | int | (int, int) | "skip"
        Baseline reference region for ``deviation`` (see :func:`deviation_tensor`).
        ``"skip"`` omits the ``deviation`` column.
    count_col : str | None
        Per-(unit, layer) cell counts, used by the ``min_baseline_cells`` gate.

    Returns
    -------
    pandas.DataFrame
        One row per (pathway, layer) with ``pathway``, ``layer``, ``score``,
        ``n_units`` and -- unless ``baseline="skip"`` -- ``deviation``.
    """
    path_names = list(path_names)
    grid = [int(L) for L in grid]
    nG, nP = len(grid), len(path_names)

    score_grid = _pool_score_grid(profiles, path_names, grid,
                                  sample_col=sample_col, layer_col=layer_col,
                                  score_agg=score_agg)

    n_units = np.zeros((nG, nP), dtype=int)
    gpos = {int(L): i for i, L in enumerate(grid)}
    for j, nm in enumerate(path_names):
        piv = profiles.pivot_table(index=sample_col, columns=layer_col, values=nm,
                                   aggfunc=score_agg)
        for L in grid:
            if L in piv.columns:
                n_units[gpos[int(L)], j] = int(np.isfinite(
                    piv[L].to_numpy(dtype=float)).sum())

    dev_grid = None
    if not (isinstance(baseline, str) and baseline == "skip"):
        vals, lays = [], []
        cnts = [] if count_col else None
        for _uid, sub in profiles.groupby(sample_col):
            sub = sub.sort_values(layer_col)
            vals.append(sub[path_names].to_numpy(dtype=float))
            lays.append(sub[layer_col].to_numpy(dtype=int))
            if cnts is not None:
                cnts.append(sub[count_col].to_numpy(dtype=float))
        D = deviation_tensor(vals, lays, grid, baseline_window=baseline,
                             min_baseline_layers=min_baseline_layers,
                             min_baseline_cells=min_baseline_cells,
                             cell_counts=cnts, verbose=verbose)
        with np.errstate(invalid="ignore"):
            dev_grid = np.nanmean(D, axis=0)

    rows = []
    for i, L in enumerate(grid):
        for j, nm in enumerate(path_names):
            rec = dict(pathway=nm, layer=int(L), score=score_grid[i, j],
                       n_units=int(n_units[i, j]))
            if dev_grid is not None:
                rec["deviation"] = float(dev_grid[i, j])
            rows.append(rec)
    return pd.DataFrame(rows)


def _pathway_deviation_tensor(profiles, path_names, grid, *, sample_col, layer_col,
                              baseline, min_baseline_layers, min_baseline_cells,
                              count_col, verbose):
    """(unit x layer x pathway) deviations, the unit ids, and the per-(unit, layer)
    cell counts, all in matching order."""
    units, vals, lays = [], [], []
    cnts = [] if count_col else None
    for uid, sub in profiles.groupby(sample_col):
        sub = sub.sort_values(layer_col)
        units.append(uid)
        vals.append(sub[path_names].to_numpy(dtype=float))
        lays.append(sub[layer_col].to_numpy(dtype=int))
        if cnts is not None:
            cnts.append(sub[count_col].to_numpy(dtype=float))
    D = deviation_tensor(vals, lays, grid, baseline_window=baseline,
                         min_baseline_layers=min_baseline_layers,
                         min_baseline_cells=min_baseline_cells,
                         cell_counts=cnts, verbose=verbose)
    gpos = {int(L): i for i, L in enumerate(grid)}
    C = np.full((len(units), len(grid)), np.nan)
    if cnts is not None:
        for u, (lay, cnt) in enumerate(zip(lays, cnts)):
            for L, n in zip(lay, cnt):
                if int(L) in gpos:
                    C[u, gpos[int(L)]] = n
    return np.asarray(units, dtype=object), D, C


def _arm_assignments(arm_vec, arms, pair_ids, n_perm, rng):
    """Every relabelling the design actually allows, or a Monte-Carlo sample of them.

    Returns ``(iterator, n_draw, exact)``. Which relabellings are legal is the whole
    question: a patient-level label may move between patients, a within-patient label
    (e.g. Pre/Post) may only swap inside its own patient. Permuting the second as if it
    were the first tests a hypothesis nobody asked.
    """
    from itertools import combinations, product
    from math import comb

    n = len(arm_vec)
    if pair_ids is None:
        idx1 = np.flatnonzero(arm_vec == arms[1])
        k, total = len(idx1), int(comb(n, len(idx1)))
        if n_perm is None or total <= n_perm:
            return (np.asarray(c, dtype=int) for c in combinations(range(n), k)), total, True
        return (rng.permutation(n)[:k] for _ in range(int(n_perm))), int(n_perm), False

    pairs = [np.flatnonzero(pair_ids == p) for p in pd.unique(pair_ids)]
    for p in pairs:
        if len(p) != 2:
            raise ValueError(
                "paired mode needs exactly two units per pair; got "
                f"{len(p)} for one pair. Drop unpaired units first.")
    order = [(p[0], p[1]) if arm_vec[p[0]] == arms[1] else (p[1], p[0]) for p in pairs]
    total = 2 ** len(pairs)

    def _from_signs(signs):
        return np.asarray([a if s else b for (a, b), s in zip(order, signs)], dtype=int)

    if n_perm is None or total <= n_perm:
        return (_from_signs(s) for s in product([True, False], repeat=len(pairs))), total, True
    return (_from_signs(rng.random(len(pairs)) < 0.5) for _ in range(int(n_perm))), \
        int(n_perm), False


def hpathway_arm_contrast(profiles, *, path_names, grid, arm_of,
                          sample_col="patient", layer_col="layer",
                          baseline="window", min_baseline_layers=3,
                          min_baseline_cells=50, count_col=None, min_cells=0,
                          pair_of=None, n_perm=None, seed=0, alpha=0.05,
                          null_quantile=0.95, null_keep=4000,
                          verbose=True):
    """Do two groups differ in where a pathway sits along the border ruler?

    The third H-Pathway channel, alongside :func:`hpathway_layer_ora` (which sets are
    border-organised) and :func:`hpathway_score_grid` (where a fixed set sits). Use it
    when the comparison is between arms -- treated vs untreated, relapse vs stable --
    rather than against a background.

    Why this rather than an enrichment test per arm
    -----------------------------------------------
    Over-representation needs genes that individually clear a significance gate, and
    supplies none of its own when they do not: with no gene above the gate the
    hypergeometric background is zero at every layer and nothing is computable, however
    the threshold is set. This test never asks about individual genes. It compares the
    arms' pooled pathway *profiles*, so a difference spread thinly over many genes --
    the case an enrichment test is least able to see -- still contributes.

    Running the enrichment test once per arm and reading the two panels side by side is
    not an alternative: it compares two *lists*, contains no contrast, and the arm with
    more units will look stronger for that reason alone.

    Parameters
    ----------
    profiles : pandas.DataFrame
        Wide per-(unit, layer) activity table, as produced by
        :func:`hplot.pathway_layer_profile` and collapsed to the independent unit.
    arm_of : Mapping | callable
        Unit -> arm label. Exactly two arms must remain after mapping.
    pair_of : Mapping | callable | None
        Unit -> pair id, for a **within-pair** label such as Pre/Post. When given, the
        null swaps arms inside each pair (2^n_pairs assignments) instead of moving
        labels between units. Leave ``None`` for a unit-level label such as patient
        outcome. This is not a tuning knob: choosing wrongly tests the wrong hypothesis.
    n_perm : int | None
        ``None`` (default) enumerates every legal assignment when that is feasible,
        which makes the p-values exact. An integer caps the work and switches to
        Monte-Carlo sampling once the exhaustive count exceeds it.
    count_col : str | None
        Per-(unit, layer) cell counts. Required by ``min_cells`` and by the
        ``min_baseline_cells`` gate.
    min_cells : int
        Minimum cells a unit must contribute to a layer for that unit-layer to be
        used. Default 0 (off). The profile is already a per-cell mean, so counts do
        not inflate the value -- but they do set its precision, and a unit with a
        handful of cells in a layer supplies an essentially random mean.
    null_quantile : float
        Quantile of the per-cell null used as the chance reference. Default 0.95.
    null_keep : int
        Cap on how many null draws are retained to estimate that quantile; draws are
        subsampled at random (unbiased) above it, so an exhaustive null of 210 or 1024
        assignments is kept in full while a large Monte-Carlo run stays bounded.

    Returns
    -------
    grid_df : pandas.DataFrame
        One row per (pathway, layer): ``gap`` (arm2 − arm1 of the baseline-centred
        profile), ``abs_gap``, ``null_ref`` (the ``null_quantile`` of |gap| under the
        null *for that same cell*), ``ratio_vs_null`` = ``abs_gap / null_ref``,
        ``p``, ``q`` (BH over the whole grid) and ``q_row`` (BH within that pathway's
        own layers, which is the family implied when each set is its own question).

        ``ratio_vs_null`` is the quantity to plot: it carries its own reference, so
        1.0 is chance level and the panel needs no external calibration. Note that
        exceeding 1.0 in a single cell is not a finding -- within one pathway, about
        one layer in twenty does so under the null by definition of the quantile.
        The per-pathway verdict is ``p_exact``, which already pays for the search
        across layers.
    summary : pandas.DataFrame
        One row per pathway: ``max_abs_gap`` (largest raw between-arm difference),
        ``max_ratio_vs_null`` (the same in units of chance), ``peak_layer`` (where
        ``ratio_vs_null`` peaks -- the raw scale would point at the noisiest layer
        instead), ``p_exact`` (max-statistic
        over layers, so the layer search is already paid for), ``q_exact``, plus the
        design facts ``n_assignments``, ``exact`` and ``p_floor`` so the resolution
        limit can be checked in code rather than read off a printout, and
        ``arm_pos`` / ``arm_neg`` naming which arm a positive ``gap`` refers to (the
        arm order is decided internally, so a caller must not guess it when labelling
        a legend).

    Notes
    -----
    **Read the resolution report before reading a row of zeros.** A permutation p-value
    cannot fall below ``1 / (n_draw + 1)``, and ``n_draw`` is fixed by the design, not by
    effort: four cases against six controls allow only 210 assignments, so no p is
    smaller than 0.0048. What that implies for a *corrected* threshold is not simply
    ``p_floor x m``: BH places the k-th smallest at ``p_floor x m / k``, so alpha is
    reachable as soon as k tests sit at the floor together -- two of 21 layers in the
    example above. The report prints that count for each family, and flags the case
    where it exceeds the number of tests available, because "nothing was significant"
    and "nothing could have been" are different findings.

    ``gap`` is signed and names which arm is higher; that is a statement about the two
    groups, not about the pathway being up- or down-regulated (a set mixes positively
    and negatively regulated members, so it has no such direction).
    """
    path_names = list(path_names)
    grid = [int(L) for L in grid]
    nG, nP = len(grid), len(path_names)
    rng = np.random.default_rng(seed)

    units, D, C = _pathway_deviation_tensor(
        profiles, path_names, grid, sample_col=sample_col, layer_col=layer_col,
        baseline=baseline, min_baseline_layers=min_baseline_layers,
        min_baseline_cells=min_baseline_cells, count_col=count_col, verbose=verbose)

    # A unit contributes a layer only when it has enough cells there to estimate a mean.
    # Without this the sparse end of the ruler dominates the raw effect purely through
    # noise: one slide with three cells in a layer moves that layer's group mean freely.
    if min_cells and count_col:
        _thin = np.isfinite(C) & (C < float(min_cells))
        if _thin.any():
            D[_thin] = np.nan
            if verbose:
                print(f"  dropped {int(_thin.sum())} of {C.size} (unit, layer) cells "
                      f"with < {min_cells} cells")

    _get = (lambda u: arm_of(u)) if callable(arm_of) else (lambda u: arm_of.get(u))
    arm_vec = np.asarray([_get(u) for u in units], dtype=object)
    keep = np.asarray([a is not None and a == a for a in arm_vec], dtype=bool)
    units, arm_vec, D = units[keep], arm_vec[keep], D[keep]
    arms = list(pd.unique(arm_vec))
    if len(arms) != 2:
        raise ValueError(f"hpathway_arm_contrast needs exactly two arms; got {arms}.")

    pair_ids = None
    if pair_of is not None:
        _pget = (lambda u: pair_of(u)) if callable(pair_of) else (lambda u: pair_of.get(u))
        pair_ids = pd.Series([_pget(u) for u in units])

    i0 = np.flatnonzero(arm_vec == arms[0])
    i1 = np.flatnonzero(arm_vec == arms[1])

    def _gap(sel1):
        sel0 = np.setdiff1d(np.arange(len(units)), sel1, assume_unique=False)
        # `_gap` is called both with non-empty `sel1` (real arm cells) and with
        # empty `sel1` (e.g. sparse layers gated out by `min_cells`, where the
        # intended signal is NaN rather than a misleading 0.0). When `sel1`
        # is empty, `np.nanmean([])` returns NaN but emits a `RuntimeWarning`
        # that `np.errstate(invalid="ignore")` cannot silence — numpy dispatches
        # empty-slice warnings through its own channel. Contain the noise
        # locally so downstream consumers (and the test report) stay clean,
        # while preserving the NaN-as-signal behaviour for sparse layers.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(D[sel1], axis=0) - np.nanmean(D[sel0], axis=0)

    obs = _gap(i1)
    obs_abs = np.abs(obs)

    it, n_draw, exact = _arm_assignments(arm_vec, arms, pair_ids, n_perm, rng)
    cnt = np.zeros((nG, nP))
    max_null = np.empty((n_draw, nP))
    obs_max = np.nanmax(np.where(np.isfinite(obs_abs), obs_abs, -np.inf), axis=0)
    # Per-cell null draws are kept so the panel can size a dot in units of chance
    # rather than raw effect. Subsampled (unbiased) when the null is large.
    keep_p = min(1.0, float(null_keep) / max(n_draw, 1))
    kept = []
    for d, sel1 in enumerate(it):
        g = np.abs(_gap(np.asarray(sel1, dtype=int)))
        cnt += (g >= obs_abs - 1e-12)
        max_null[d] = np.nanmax(np.where(np.isfinite(g), g, -np.inf), axis=0)
        if keep_p >= 1.0 or rng.random() < keep_p:
            kept.append(g.astype(np.float32))
    # Layers gated out by `min_cells` are all-NaN by design; NaN is the signal.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        null_ref = (np.nanpercentile(np.stack(kept), 100.0 * null_quantile, axis=0)
                    if kept else np.full((nG, nP), np.nan))

    p = (1.0 + cnt) / (n_draw + 1.0)
    p = np.where(np.isfinite(obs_abs), p, np.nan)
    q = np.full(nG * nP, np.nan)
    flat = p.ravel()
    m = np.isfinite(flat)
    q[m] = benjamini_hochberg(flat[m])
    q = q.reshape(nG, nP)

    # Per-pathway BH across that pathway's own layers. This is the family the design
    # implies when each set is a separate pre-specified question, and it is far less
    # conservative than correcting across the whole grid. Adjacent layers are positively
    # correlated, under which BH remains valid.
    q_row = np.full((nG, nP), np.nan)
    for j in range(nP):
        col = p[:, j]
        ok = np.isfinite(col)
        if ok.any():
            q_row[ok, j] = benjamini_hochberg(col[ok])

    grid_df = pd.DataFrame([
        dict(pathway=nm, layer=int(L), gap=float(obs[i, j]),
             abs_gap=float(obs_abs[i, j]), null_ref=float(null_ref[i, j]),
             ratio_vs_null=float(obs_abs[i, j] / null_ref[i, j])
             if np.isfinite(null_ref[i, j]) and null_ref[i, j] > 0 else np.nan,
             p=float(p[i, j]), q=float(q[i, j]), q_row=float(q_row[i, j]))
        for i, L in enumerate(grid) for j, nm in enumerate(path_names)])

    p_exact = (1.0 + (max_null >= obs_max[None, :] - 1e-12).sum(axis=0)) / (n_draw + 1.0)
    # The peak is read off the chance-standardised profile. On the raw scale it would
    # simply mark the noisiest layer -- typically the sparse end of the ruler, where a
    # unit contributing a handful of cells inflates the group mean.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.isfinite(null_ref) & (null_ref > 0), obs_abs / null_ref, np.nan)
    peak = [int(grid[int(np.nanargmax(np.where(np.isfinite(ratio[:, j]),
                                               ratio[:, j], -np.inf)))])
            for j in range(nP)]
    max_ratio = np.nanmax(np.where(np.isfinite(ratio), ratio, -np.inf), axis=0)
    summary = pd.DataFrame(dict(
        pathway=path_names, max_abs_gap=obs_max, max_ratio_vs_null=max_ratio,
        peak_layer=peak, p_exact=p_exact)).assign(
        q_exact=lambda d: benjamini_hochberg(d["p_exact"].to_numpy()),
        arm_pos=str(arms[1]), arm_neg=str(arms[0]),
        n_assignments=int(n_draw), exact=bool(exact),
        p_floor=1.0 / (n_draw + 1.0),
    ).sort_values("p_exact", kind="mergesort").reset_index(drop=True)

    p_floor = 1.0 / (n_draw + 1.0)
    m_tested = int(np.isfinite(p).sum())

    def _k_min(m):
        # BH puts the k-th smallest at p_floor * m / k, so alpha becomes reachable once
        # this many tests sit at the floor together. Quoting p_floor * m alone describes
        # only the case where a single test is extreme, and understates BH badly.
        return int(np.floor(p_floor * m / alpha)) + 1

    if verbose:
        print(f"arm contrast {arms[1]} - {arms[0]}: {len(i1)} vs {len(i0)} units | "
              f"{'EXHAUSTIVE' if exact else 'Monte-Carlo'} null, {n_draw} assignments "
              f"({'within-pair swap' if pair_ids is not None else 'unit relabel'})")
        print(f"  smallest attainable p = {p_floor:.4g}")
        print(f"  BH reaches q<{alpha:g} once this many tests sit at that floor "
              f"together: {_k_min(nG)} of {nG} layers within a pathway (q_row) | "
              f"{_k_min(nP)} of {nP} pathways (q_exact) | "
              f"{_k_min(m_tested)} of {m_tested} cells (q, whole grid)")
        _unreach = [nm for nm, m in (("q_row", nG), ("q_exact", nP),
                                     ("q (whole grid)", m_tested)) if _k_min(m) > m]
        if _unreach:
            print(f"  *** out of reach at this design (would need more tests at the "
                  f"floor than exist): {', '.join(_unreach)}")
        print(f"  observed: q_row<{alpha:g} in {int((q_row < alpha).sum())} cells | "
              f"p_exact<{alpha:g} in {int((summary['p_exact'] < alpha).sum())} of {nP} "
              f"pathways (chance ~{alpha * nP:.1f}) | "
              f"q<{alpha:g} in {int((q < alpha).sum())} cells")
    return grid_df, summary
