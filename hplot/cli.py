"""hplot command-line interface.

Sub-commands
------------
hplot plot   — draw H-Plot curves from a CSV (batch-safe)
hplot test   — per-layer Mann-Whitney / cluster-mass permutation test
hplot gam    — Stage-2 GAM effect size with optional confounder adjustment
hplot screen — multi-feature cluster-mass border-gradient screen -> ranking CSV
hplot loci   — render an H-Loci Summary panel from a ranking CSV

Usage examples
--------------
::

    hplot plot  -i data.csv --target immune_fraction --group hpv_status -o out/

    hplot test  -i data.csv --target immune_fraction --group hpv_status \\
                --permutations 999 -o out/pvalues.csv

    hplot gam   -i data.csv --target immune_fraction --group hpv_status \\
                --at-layer 0 --covariates AGE late_stage is_female

"""

import argparse
import sys
import os
import numpy as np
import pandas as pd


def _out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ── plot ──────────────────────────────────────────────────────────────────

def _cmd_plot(args):
    from hplot.runners import run_hplot_batch
    df = pd.read_csv(args.input)
    run_hplot_batch(
        df=df,
        targets=args.targets,
        layer=args.layer,
        group=args.group,
        distance=args.distance,
        unit=args.unit,
        output=_out_dir(args.output),
        prefix=args.prefix,
        ci_show=args.ci,
        format=args.format,
        dpi=args.dpi,
        exclude_base=args.exclude_base,
        min_base_excluded_count=args.min_base_excluded_count,
    )
    print(f"[hplot plot]  figures written to {args.output}/")


def _add_plot_parser(sub):
    p = sub.add_parser(
        "plot",
        help="Draw H-Plot curves and save as SVG/PNG/PDF.",
        description="Fit per-layer means +/- CI for each group and produce H-Plot figures.",
    )
    p.add_argument("-i", "--input", required=True, help="Input CSV file.")
    p.add_argument("--targets", nargs="+", default=["target_prop"],
                   help="Column name(s) for the target quantity.")
    p.add_argument("--layer",    default="layer",  help="Layer index column.")
    p.add_argument("--group",    default=None,      help="Group label column.")
    p.add_argument("--distance", default=None,      help="Physical distance column.")
    p.add_argument("-u", "--unit", default=None,    help="Distance unit label (e.g. um).")
    p.add_argument("-o", "--output", default="hplots", help="Output directory.")
    p.add_argument("-p", "--prefix", default="hplot",  help="Output filename prefix.")
    p.add_argument("-f", "--format", default="svg",
                   choices=["svg", "pdf", "png"],    help="Output image format.")
    p.add_argument("--dpi", type=int, default=300,   help="DPI for PNG output.")
    p.add_argument("--ci", action="store_true",      help="Show confidence interval bands.")
    p.add_argument("--exclude-base", dest="exclude_base", action="store_true",
                   help="Exclude base cells from the denominator: "
                        "target_count / (all_count - base_count).")
    p.add_argument("--min-base-excluded-count", dest="min_base_excluded_count",
                   type=int, default=1,
                   help="Drop layers where all_count - base_count < this "
                        "(only with --exclude-base; default 1).")
    p.set_defaults(func=_cmd_plot)


# ── test ──────────────────────────────────────────────────────────────────

def _cmd_test(args):
    from hplot.stats import compute_layer_pvalues
    df = pd.read_csv(args.input)
    target = args.target
    if args.exclude_base:
        from hplot.runners import add_base_excluded_proportion
        df, target = add_base_excluded_proportion(
            df, min_base_excluded_count=args.min_base_excluded_count)
    elif target is None:
        raise SystemExit(
            "hplot test: --target is required unless --exclude-base is given.")
    pvals = compute_layer_pvalues(
        df,
        prop=target,
        layer_col=args.layer,
        group_col=args.group,
        groups=tuple(args.groups) if args.groups else None,
        test=args.test,
        distance_col=args.distance,
        min_n=args.min_n,
        correction=args.correction,
    )
    if args.output:
        pvals.to_csv(args.output, index=False)
        print(f"[hplot test]  p-value table written to {args.output}")
    else:
        print(pvals.to_string(index=False))

    if args.permutations > 0:
        _cluster_mass_summary(df, pvals, args, target)


def _cluster_mass_summary(df, pvals, args, target):
    from hplot.stats import compute_layer_pvalues
    col = "p_adj" if args.correction else "p_value"
    sig = pvals[pvals[col] < args.threshold]
    if sig.empty:
        print(f"[hplot test]  No layers significant at p < {args.threshold}; "
              "cluster-mass not computed.")
        return
    obs_mass = (args.threshold - sig[col]).clip(lower=0).sum()
    rng = np.random.default_rng(args.seed)
    null_masses = []
    for _ in range(args.permutations):
        df_perm = df.copy()
        g_vals = df_perm[args.group].to_numpy()
        rng.shuffle(g_vals)
        df_perm[args.group] = g_vals
        try:
            pv_perm = compute_layer_pvalues(
                df_perm, prop=target, layer_col=args.layer,
                group_col=args.group,
                groups=tuple(args.groups) if args.groups else None,
                test=args.test, distance_col=args.distance,
                min_n=args.min_n, correction=args.correction,
            )
            sig_p = pv_perm[pv_perm[col] < args.threshold]
            null_masses.append((args.threshold - sig_p[col]).clip(lower=0).sum())
        except Exception:
            null_masses.append(0.0)
    perm_p = float((np.array(null_masses) >= obs_mass).mean())
    print(f"[hplot test]  Cluster-mass: obs={obs_mass:.4f}  "
          f"perm-p={perm_p:.4f}  (n_perm={args.permutations},"
          f" threshold={args.threshold})")


def _add_test_parser(sub):
    p = sub.add_parser(
        "test",
        help="Per-layer Mann-Whitney test + optional cluster-mass permutation.",
        description=(
            "Compute per-layer p-values between two groups.  "
            "Optionally runs a cluster-mass permutation test to control the "
            "family-wise error rate across the layer dimension."
        ),
    )
    p.add_argument("-i", "--input",   required=True, help="Input CSV file.")
    p.add_argument("--target",        default=None,
                   help="Target proportion column (required unless --exclude-base).")
    p.add_argument("--layer",         default="layer", help="Layer index column.")
    p.add_argument("--group",         required=True,   help="Group label column.")
    p.add_argument("--groups", nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="Explicit group pair (required when >2 unique values).")
    p.add_argument("--distance",      default=None, help="Physical distance column.")
    p.add_argument("--test",          default="mannwhitney",
                   choices=["mannwhitney", "ttest", "welch"],
                   help="Per-layer statistical test (default: mannwhitney).")
    p.add_argument("--correction",    default=None,
                   choices=["bonferroni", "fdr_bh"],
                   help="Multiple-testing correction across layers.")
    p.add_argument("--min-n", dest="min_n", type=int, default=3,
                   help="Min cases per group to test a layer (default 3).")
    p.add_argument("--permutations",  type=int, default=0,
                   help="Label-permutations for cluster-mass test (0 = skip).")
    p.add_argument("--threshold",     type=float, default=0.05,
                   help="Per-layer significance threshold for cluster-mass.")
    p.add_argument("--seed",          type=int, default=42, help="Random seed.")
    p.add_argument("--exclude-base", dest="exclude_base", action="store_true",
                   help="Derive the target from counts as "
                        "target_count / (all_count - base_count) before testing.")
    p.add_argument("--min-base-excluded-count", dest="min_base_excluded_count",
                   type=int, default=1,
                   help="Drop layers where all_count - base_count < this "
                        "(only with --exclude-base; default 1).")
    p.add_argument("-o", "--output",  default=None,
                   help="Output CSV path for p-value table (stdout if omitted).")
    p.set_defaults(func=_cmd_test)


# ── gam ───────────────────────────────────────────────────────────────────

def _cmd_gam(args):
    from hplot.stats import gam_pooled_effect, gam_group_curves
    df = pd.read_csv(args.input)
    target = args.target
    if args.exclude_base:
        from hplot.runners import add_base_excluded_proportion
        df, target = add_base_excluded_proportion(
            df, min_base_excluded_count=args.min_base_excluded_count)
    elif target is None:
        raise SystemExit(
            "hplot gam: --target is required unless --exclude-base is given.")
    covariates = args.covariates or None
    effect, pval, n = gam_pooled_effect(
        long_df=df,
        target_col=target,
        layer_col=args.layer,
        group_col=args.group,
        at_layer=args.at_layer,
        groups=tuple(args.groups) if args.groups else None,
        covariate_cols=covariates,
        n_splines=args.n_splines,
    )
    cov_str = ", ".join(covariates) if covariates else "none"
    print(f"[hplot gam]  target={target}  group={args.group}  "
          f"at_layer={args.at_layer}")
    print(f"             covariates  : [{cov_str}]")
    print(f"             effect (high - low) = {effect:+.4f}")
    print(f"             p-value (group term) = {pval:.3e}   n = {n}")
    if args.curves_output:
        grid = np.arange(df[args.layer].min(), df[args.layer].max() + 1)
        curves = gam_group_curves(
            long_df=df,
            target_col=target,
            layer_col=args.layer,
            group_col=args.group,
            grid=grid,
            groups=tuple(args.groups) if args.groups else None,
            n_splines=args.n_splines,
        )
        rows = []
        for grp, (pred, ci) in curves.items():
            for i, lyr in enumerate(grid):
                rows.append({"group": grp, "layer": lyr,
                             "gam_pred": pred[i],
                             "ci_lower": ci[i, 0], "ci_upper": ci[i, 1]})
        pd.DataFrame(rows).to_csv(args.curves_output, index=False)
        print(f"[hplot gam]  curve predictions written to {args.curves_output}")


def _add_gam_parser(sub):
    p = sub.add_parser(
        "gam",
        help="Stage-2 GAM effect size with optional confounder adjustment.",
        description=(
            "Fit target ~ s(layer) + group [+ covariates] using a penalised "
            "B-spline GAM and report the high-minus-low group difference at "
            "--at-layer together with the Wald p-value of the group term."
        ),
    )
    p.add_argument("-i", "--input",  required=True, help="Input CSV file.")
    p.add_argument("--target",       default=None,
                   help="Response column (required unless --exclude-base).")
    p.add_argument("--layer",        default="layer", help="Layer index column.")
    p.add_argument("--group",        required=True,   help="Group label column.")
    p.add_argument("--groups", nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="Explicit (low, high) group pair.")
    p.add_argument("--at-layer", dest="at_layer", type=float, required=True,
                   help="Layer at which to evaluate the group effect.")
    p.add_argument("--covariates", nargs="+", default=None, metavar="COL",
                   help="Columns to include as linear confounders.")
    p.add_argument("--n-splines", dest="n_splines", type=int, default=10,
                   help="Number of B-spline basis functions (default 10).")
    p.add_argument("--exclude-base", dest="exclude_base", action="store_true",
                   help="Derive the response from counts as "
                        "target_count / (all_count - base_count) before fitting.")
    p.add_argument("--min-base-excluded-count", dest="min_base_excluded_count",
                   type=int, default=1,
                   help="Drop layers where all_count - base_count < this "
                        "(only with --exclude-base; default 1).")
    p.add_argument("--curves-output", dest="curves_output", default=None,
                   help="CSV path to save per-group GAM predictions + 95%% CI.")
    p.set_defaults(func=_cmd_gam)


# ── screen / loci helpers ──────────────────────────────────────────────────

def _pivot_slides(df, sample_col, layer_col, unit_col, value_col):
    """Long CSV -> (values, layers, unit_names) for ``deviation_tensor``.

    Returns one ``(n_layers_slide, n_units)`` value matrix and matching integer
    layer vector per slide, with a shared unit ordering across all slides.
    """
    units = sorted(df[unit_col].astype(str).unique())
    values, layers = [], []
    for _sid, g in df.groupby(sample_col, sort=True):
        piv = (g.pivot_table(index=layer_col, columns=unit_col, values=value_col,
                             aggfunc="mean")
                .reindex(columns=units))
        piv = piv.sort_index()
        values.append(piv.to_numpy(dtype=float))
        layers.append(piv.index.to_numpy().astype(int))
    return values, layers, units


def _run_screen_from_csv(args):
    """Shared screen driver used by ``hplot screen`` and ``hplot loci --screen``.

    Returns ``(long_df, wide_df, layer_um)``.
    """
    from hplot.stats import deviation_tensor, gradient_cluster_mass_screen
    df = pd.read_csv(args.input)
    values, layers, units = _pivot_slides(
        df, args.sample, args.layer, args.unit, args.value)

    if args.grid:
        grid = np.arange(int(args.grid[0]), int(args.grid[1]) + 1)
    else:
        allL = np.concatenate([lay for lay in layers if lay.size])
        grid = np.arange(int(allL.min()), int(allL.max()) + 1)

    baseline = args.baseline
    if baseline not in ("window", "far", "core"):
        a, b = baseline.split(",")
        baseline = (int(a), int(b))

    layer_um = None
    if args.distance:
        acc = {}
        for _L, _d in zip(df[args.layer].to_numpy().astype(int),
                          df[args.distance].to_numpy(dtype=float)):
            acc.setdefault(int(_L), []).append(float(_d))
        layer_um = {L: float(np.mean(v)) for L, v in acc.items()}

    D = deviation_tensor(values, layers, grid, baseline_window=baseline,
                         min_baseline_layers=args.min_baseline_layers)
    res = gradient_cluster_mass_screen(
        D, grid, unit_names=units, band_mode=args.band_mode,
        cluster_alpha=args.cluster_alpha, min_w=args.min_w,
        min_per_group=args.min_per_group, n_perm=args.permutations,
        seed=args.seed, layer_um=layer_um, progress=args.progress,
    )
    return res["long"], res["wide"], layer_um


def _add_screen_args(p):
    """Column / screen options shared by ``screen`` and ``loci --screen``."""
    p.add_argument("--sample", default="sample", help="Slide/sample id column.")
    p.add_argument("--layer",  default="layer",  help="Signed layer index column.")
    p.add_argument("--unit",   default="unit",
                   help="Feature column (gene / LR pair / cell type).")
    p.add_argument("--value",  default="value",  help="Per-layer value column.")
    p.add_argument("--distance", default=None,
                   help="Physical-distance (µm) column; enables *_um outputs.")
    p.add_argument("--grid", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                   help="Analysis-window layer range (default: data min..max).")
    p.add_argument("--baseline", default="window",
                   help="Baseline region: window | far | core | 'a,b'.")
    p.add_argument("--min-baseline-layers", dest="min_baseline_layers",
                   type=int, default=3,
                   help="Min baseline-region layers per slide (default 3).")
    p.add_argument("--band-mode", dest="band_mode", default="dominant",
                   choices=["dominant", "bidirectional"],
                   help="Winner-take-all (dominant) or per-direction bands.")
    p.add_argument("--cluster-alpha", dest="cluster_alpha", type=float, default=0.05,
                   help="Cluster-forming alpha (chi2 threshold; default 0.05).")
    p.add_argument("--min-w", dest="min_w", type=int, default=1,
                   help="Minimum contiguous band width in layers (default 1).")
    p.add_argument("--min-per-group", dest="min_per_group", type=int, default=10,
                   help="Minimum contributing slides per layer (default 10).")
    p.add_argument("--permutations", type=int, default=1000,
                   help="Layer-shuffle permutations (default 1000).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    p.add_argument("--progress", action="store_true",
                   help="Show a tqdm bar over permutations.")


# ── screen ─────────────────────────────────────────────────────────────────

def _cmd_screen(args):
    long_df, wide_df, _ = _run_screen_from_csv(args)
    long_df.to_csv(args.output, index=False)
    n_band = len(long_df)
    print(f"[hplot screen]  ranking table ({n_band} banded rows) -> {args.output}")
    if args.wide_output is not None:
        wide_df.to_csv(args.wide_output, index=False)
        print(f"[hplot screen]  wide table -> {args.wide_output}")


def _add_screen_parser(sub):
    p = sub.add_parser(
        "screen",
        help="Multi-feature cluster-mass border-gradient screen -> ranking CSV.",
        description=(
            "Run gradient_cluster_mass_screen() across every feature in a long "
            "CSV (sample x layer x unit x value) and write the banded ranking "
            "table consumed by 'hplot loci'."
        ),
    )
    p.add_argument("-i", "--input", required=True,
                   help="Long CSV: sample, layer, unit, value columns.")
    _add_screen_args(p)
    p.add_argument("-o", "--output", default="ranking.csv",
                   help="Output ranking CSV (one row per banded feature).")
    p.add_argument("--wide-output", dest="wide_output", default=None,
                   help="Optional CSV for the per-feature wide table.")
    p.set_defaults(func=_cmd_screen)


# ── loci ───────────────────────────────────────────────────────────────────

def _layer_um_from_ranking(df):
    """Reconstruct a ``{layer L: physical distance µm}`` map from a ranking
    table using its (layer, µm) column pairs, so 'hplot loci' can draw the
    same dual x-axis as the H-Plot curves without re-reading the raw data.
    Returns ``None`` when no usable pair is present."""
    import hplot
    pairs = [("band_start_layer", "band_start_um"),
             ("band_end_layer", "band_end_um"),
             ("center_layer", "center_um"),
             ("peak_layer", "peak_um")]
    layers, dists = [], []
    for lcol, ucol in pairs:
        if lcol in df.columns and ucol in df.columns:
            layers.append(pd.to_numeric(df[lcol], errors="coerce").to_numpy())
            dists.append(pd.to_numeric(df[ucol], errors="coerce").to_numpy())
    if not layers:
        return None
    L = np.concatenate(layers)
    U = np.concatenate(dists)
    ok = np.isfinite(L) & np.isfinite(U)
    if not ok.any():
        return None
    return hplot.build_layer_distance_map(L[ok], U[ok])


def _cmd_loci(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hplot

    if args.screen:
        long_df, wide_df, layer_um = _run_screen_from_csv(args)
        df = wide_df if args.kind == "bidirectional" else long_df
    else:
        df = pd.read_csv(args.input)
        layer_um = _layer_um_from_ranking(df)

    if args.fdr_col and args.fdr_max is not None and args.fdr_col in df.columns:
        df = df[df[args.fdr_col] <= args.fdr_max]
    if args.top_n and args.mass_col in df.columns:
        df = df.sort_values(args.mass_col, ascending=False).head(args.top_n)
    if df.empty:
        raise SystemExit("hplot loci: no rows to plot after filtering.")

    sort = None if args.sort == "none" else args.sort
    n = len(df)
    fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 24.0))
    fig, ax = plt.subplots(figsize=(args.width, fig_h))

    if args.kind == "bands":
        hplot.plot_hloci_bands(
            df[args.lo_col], df[args.hi_col], df[args.dir_col],
            peak=df[args.peak_col] if args.peak_col in df.columns else None,
            mass=df[args.mass_col] if args.mass_col in df.columns else None,
            labels=df[args.label_col], sort=sort, ax=ax,
            xlabel="border layer L", title=args.title)
    elif args.kind == "summary":
        hplot.plot_hloci_summary(
            df[args.peak_col], df[args.dir_col],
            weights=df[args.mass_col] if args.mass_col in df.columns else None,
            labels=df[args.label_col], ax=ax,
            xlabel="border layer L", title=args.title)
    else:  # bidirectional (wide schema)
        hplot.plot_hloci_bands_bidirectional(
            df[args.label_col],
            df["elevated_start"], df["elevated_end"],
            df["depressed_start"], df["depressed_end"],
            elev_center=df.get("elevated_center"),
            depr_center=df.get("depressed_center"),
            elev_mass=df.get("elevated_mass"),
            depr_mass=df.get("depressed_mass"),
            sort_by=None if sort is None else "dominant_center",
            ax=ax, title=args.title)

    if layer_um is not None:
        hplot.add_border_distance_axis(ax, layer_um)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"[hplot loci]  {args.kind} panel ({n} rows) -> {args.output}")


def _add_loci_parser(sub):
    p = sub.add_parser(
        "loci",
        help="Render an H-Loci Summary panel from a ranking CSV.",
        description=(
            "Draw an H-Loci Summary from a ranking table (e.g. 'hplot screen' "
            "output). The canonical 'bands' view draws each feature as a "
            "horizontal band coloured by direction (up_color=elevated, "
            "down_color=depressed) with a vertical tick at the cluster-mass "
            "peak; 'bidirectional' splits elevated/depressed bars per row; "
            "'summary' is the legacy strip+triangle view. Pass --screen to run "
            "the screen first from a raw long CSV."),
    )
    p.add_argument("-i", "--input", required=True,
                   help="Ranking CSV (or raw long CSV when --screen is set).")
    p.add_argument("-o", "--output", default="hloci.svg",
                   help="Output figure path (.svg/.pdf/.png).")
    p.add_argument("--kind", default="bands",
                   choices=["bands", "summary", "bidirectional"],
                   help="Panel style: bands (default, canonical band view) | "
                        "bidirectional | summary (legacy strip+triangle).")
    p.add_argument("--sort", default="outer_to_inner",
                   choices=["outer_to_inner", "inner_to_outer", "none"],
                   help="Row ordering by band centre (default outer_to_inner).")
    p.add_argument("--top-n", dest="top_n", type=int, default=None,
                   help="Keep the top-N rows by cluster mass before drawing.")
    p.add_argument("--width", type=float, default=6.4, help="Figure width (in).")
    p.add_argument("--dpi", type=int, default=300, help="Raster DPI (default 300).")
    p.add_argument("--title", default=None, help="Panel title.")
    # ranking-table column names (defaults match 'hplot screen' output)
    p.add_argument("--label-col", dest="label_col", default="gene")
    p.add_argument("--lo-col",   dest="lo_col",   default="band_start_layer")
    p.add_argument("--hi-col",   dest="hi_col",   default="band_end_layer")
    p.add_argument("--dir-col",  dest="dir_col",  default="direction")
    p.add_argument("--peak-col", dest="peak_col", default="peak_layer")
    p.add_argument("--mass-col", dest="mass_col", default="cluster_mass")
    p.add_argument("--fdr-col",  dest="fdr_col",  default="fdr")
    p.add_argument("--fdr-max",  dest="fdr_max",  type=float, default=None,
                   help="Drop rows with FDR above this before drawing.")
    # optional: chain the screen from a raw long CSV
    p.add_argument("--screen", action="store_true",
                   help="Run 'hplot screen' first (input is a raw long CSV).")
    _add_screen_args(p)
    p.set_defaults(func=_cmd_loci)


# ── entry point ───────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="hplot",
        description="H-Plot: graph-geodesic spatial profiling at tissue boundaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Sub-commands:",
            "  plot   Draw H-Plot curves from a CSV.",
            "  test   Per-layer Mann-Whitney + cluster-mass permutation test.",
            "  gam    Stage-2 GAM effect size with confounder adjustment.",
            "  screen Multi-feature cluster-mass screen -> ranking CSV.",
            "  loci   Render an H-Loci Summary panel from a ranking CSV.",
            "",
            "Run  hplot <sub-command> --help  for full options.",
        ]),
    )
    sub = parser.add_subparsers(dest="command")
    _add_plot_parser(sub)
    _add_test_parser(sub)
    _add_gam_parser(sub)
    _add_screen_parser(sub)
    _add_loci_parser(sub)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
