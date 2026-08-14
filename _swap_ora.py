from pathlib import Path

P = Path("/workspace/wsinsight/hplot/hplot/stats.py")
lines = P.read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(lines) if l.startswith("def _matched_draws"))
print(f"truncating at line {cut + 1}: {lines[cut].rstrip()!r}")
head = "".join(lines[:cut]).rstrip("\n") + "\n"

NEW = '''

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
'''
P.write_text(head + NEW)
print(f"rewrote {P}: {len(head.splitlines())} -> {len((head + NEW).splitlines())} lines")
