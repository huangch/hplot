import pandas as pd
import numpy as np
from scipy.stats import t, norm, mannwhitneyu, ttest_ind, chi2, rankdata, kruskal

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
