import pandas as pd
import numpy as np
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


def deviation_tensor(values, layers, grid, *, baseline_window=None,
                     min_baseline_layers=3, verbose=True):
    """Assemble the per-slide deviation tensor for the gradient screen.

    Each slide's per-layer values are placed onto a common ``grid`` (the
    cluster-mass analysis window) after subtracting a per-slide, per-unit
    **baseline**. The baseline *reference region* is selected with
    ``baseline_window`` and is independent of the layers that are actually
    tested for bands (those are always the ``grid`` layers):

    * ``None`` or ``"window"`` — baseline = mean over the slide's layers that
      fall inside ``grid`` (the historical default; deviations are relative to
      the slide's own window-wide level);
    * ``"far"`` — baseline = mean over the slide's layers beyond the
      outer end of ``grid`` (``L > max(grid)``), i.e. the distal far-field
      away from the base object (a "resting tissue" reference);
    * ``"core"`` — baseline = mean over the slide's layers beyond the
      interior end of ``grid`` (``L < min(grid)``), i.e. deep inside the
      base object;
    * ``(a, b)`` — baseline = mean over the slide's layers with
      ``a <= L <= b`` (an explicit reference window).

    A slide that contributes fewer than ``min_baseline_layers`` layers to its
    chosen baseline region is dropped from the tensor (its whole row is NaN),
    so an unstable / poorly sampled reference never enters the pooled statistic.

    Parameters
    ----------
    values : sequence of ndarray, each ``(n_layers_slide, n_units)``
        Per-slide per-layer values (e.g. mean expression).
    layers : sequence of int ndarray, each ``(n_layers_slide,)``
        Integer layer coordinate for every row of the matching ``values``.
    grid : array-like, ``(n_layers,)``
        Layer coordinates of the analysis window (the tested layers).
    baseline_window : None | "window" | "far" | "core" | (int, int)
        Baseline reference-region selector (see above). Default ``None``
        (self-centre over the analysis window).
    min_baseline_layers : int
        Minimum number of baseline-region layers a slide must contribute; a
        slide below this is skipped (all-NaN). Default 3 (a 3-layer average
        roughly halves the baseline noise vs a single layer). The default
        ``"window"`` baseline is almost always well above this, so raising it
        mainly guards the sparser ``"far"`` / ``"core"`` references.
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

    def _baseline_mask(lay):
        if bw is None or bw == "window":
            return np.array([int(L) in gpos for L in lay], dtype=bool)
        if bw == "far":
            return lay > hi
        if bw == "core":
            return lay < lo
        try:
            a, b = bw
        except (TypeError, ValueError):
            raise ValueError(
                "baseline_window must be None, 'window', 'far', 'core', "
                "or an (a, b) tuple.")
        return (lay >= int(a)) & (lay <= int(b))

    D = np.full((n_slides, nG, n_units), np.nan, dtype=float)
    n_data = 0
    n_skip_base = 0
    for si, (V, lay) in enumerate(zip(values, layers)):
        V = np.asarray(V, dtype=float)
        lay = np.asarray(lay).astype(int)
        if V.size == 0 or lay.size == 0:
            continue
        n_data += 1
        bmask = _baseline_mask(lay)
        if int(np.count_nonzero(bmask)) < min_baseline_layers:
            n_skip_base += 1  # unstable / missing baseline -> skip this slide
            continue
        base = V[bmask].mean(axis=0)
        for k, L in enumerate(lay):
            gi = gpos.get(int(L))
            if gi is not None:
                D[si, gi] = V[k] - base
    if verbose and n_skip_base:
        region = bw if bw is not None else "window"
        print(f"\u26a0 deviation_tensor: skipped {n_skip_base}/{n_data} slide(s) "
              f"with < {min_baseline_layers} baseline layer(s) in {region!r} "
              f"region (excluded from the pooled statistic).")
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

    The permutation null preserves the analysis's original scheme: within each
    slide the layers of the deviation tensor are shuffled, breaking the
    layer<->value association while keeping the per-slide marginal.

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
        Number of layer-shuffle permutations. Default 1000.
    seed : int
        RNG seed. Default 0.
    layer_um : dict | None
        Optional ``{layer: micron}`` map to populate the ``*_um`` columns.
    progress : bool
        Show a tqdm bar over permutations if available. Default False.

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

    def _perm_iter():
        it = range(n_perm)
        if progress:
            try:
                from tqdm.auto import tqdm as _tqdm
                return _tqdm(it, desc="perms", leave=False)
            except ImportError:
                return it
        return it

    rng = np.random.default_rng(seed)

    if band_mode == "dominant":
        # ---- observed: combined winner-take-all band per unit ---------------
        obs_mass = np.zeros(n_units)
        bs_o = np.full(n_units, -1)
        be_o = np.full(n_units, -1)
        for u in range(n_units):
            m, bs, be = _best_band_combined(h_obs[:, u], h_obs[:, u] > thr, min_w)
            obs_mass[u], bs_o[u], be_o[u] = m, bs, be
        # ---- pooled-permutation null (historical estimator) -----------------
        null = np.empty((n_perm, n_units), dtype=np.float32)
        for b in _perm_iter():
            Dp = np.empty_like(D)
            for si in range(n_slides):
                Dp[si] = D[si][rng.permutation(n_layers)]
            hp = np.nan_to_num(_signed_layer_z(Dp, min_per_group) ** 2, nan=0.0)
            for u in range(n_units):
                null[b, u], _, _ = _best_band_combined(hp[:, u], hp[:, u] > thr, min_w)
        perm_p = np.maximum((null >= obs_mass[None, :]).mean(0), 1.0 / n_perm)
        order = np.argsort(-obs_mass)
        obs_s = obs_mass[order]
        flat = np.sort(null.ravel())
        ge = flat.size - np.searchsorted(flat, obs_s, side="left")
        EV = ge / n_perm
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
        return dict(thr=thr, z=z_obs, long=long_df, wide=wide_df, band_mode=band_mode)

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

    null_pos = np.zeros((n_perm, n_units), dtype=np.float32)
    null_neg = np.zeros((n_perm, n_units), dtype=np.float32)
    for b in _perm_iter():
        Dp = np.empty_like(D)
        for si in range(n_slides):
            Dp[si] = D[si][rng.permutation(n_layers)]
        zp = _signed_layer_z(Dp, min_per_group)
        hp = np.nan_to_num(zp ** 2, nan=0.0)
        supra = hp > thr
        pos = supra & (zp > 0)
        neg = supra & (zp < 0)
        for u in range(n_units):
            null_pos[b, u], _, _ = _best_band_masked(hp[:, u], pos[:, u], min_w)
            null_neg[b, u], _, _ = _best_band_masked(hp[:, u], neg[:, u], min_w)

    # plus-one directional permutation p-values (never zero)
    p_pos = (1.0 + (null_pos >= obs_pos[None, :]).sum(0)) / (n_perm + 1.0)
    p_neg = (1.0 + (null_neg >= obs_neg[None, :]).sum(0)) / (n_perm + 1.0)
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
    return dict(thr=thr, z=z_obs, long=long_df, wide=wide_df, band_mode=band_mode)


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
                        baseline_window="far", min_baseline_layers=3,
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
    for _sid, sub in profiles.groupby(sample_col):
        sub = sub.sort_values(layer_col)
        vals.append(sub[path_names].to_numpy(dtype=float))
        lays.append(sub[layer_col].to_numpy(dtype=int))
    Ddev = deviation_tensor(vals, lays, grid, baseline_window=baseline_window,
                            min_baseline_layers=min_baseline_layers,
                            verbose=verbose)
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
                          score_agg="median", deviation="far",
                          min_baseline_layers=3, min_per_group=3,
                          deviation_alternative="two-sided",
                          contrasts=None, long_df=None,
                          value_col="score", pathway_col="pathway",
                          verbose=True):
    """Build the tidy (pathway x layer) grid that feeds ``plot_hpathway_dotplot``.

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
    deviation : None | "window" | "far" | "core" | (int, int)
        Baseline reference-region selector for the deviation FDR
        (see :func:`deviation_tensor`). ``None`` skips ``fdr_dev``.
    min_baseline_layers, min_per_group : int
        Deviation baseline floor and minimum non-NaN samples per test.
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

    Returns
    -------
    pandas.DataFrame
        Long grid with columns ``pathway, layer, score`` (+ ``fdr_dev`` and
        ``p_<name>``/``fdr_<name>`` per contrast), one row per (pathway, layer).
    """
    path_names = list(path_names)
    grid = [int(L) for L in grid]
    nG, nP = len(grid), len(path_names)

    score_grid = _pool_score_grid(profiles, path_names, grid,
                                   sample_col=sample_col, layer_col=layer_col,
                                   score_agg=score_agg)

    if deviation is not None:
        fdr_dev, dir_dev = _deviation_fdr_grid(
            profiles, path_names, grid, sample_col=sample_col,
            layer_col=layer_col, baseline_window=deviation,
            min_baseline_layers=min_baseline_layers, min_per_group=min_per_group,
            alternative=deviation_alternative, verbose=verbose)
    else:
        fdr_dev = dir_dev = None

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
