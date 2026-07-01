"""hplot command-line interface.

Sub-commands
------------
hplot plot   — draw H-Plot curves from a CSV (batch-safe)
hplot test   — per-layer Mann-Whitney / cluster-mass permutation test
hplot gam    — Stage-2 GAM effect size with optional confounder adjustment

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
    p.set_defaults(func=_cmd_plot)


# ── test ──────────────────────────────────────────────────────────────────

def _cmd_test(args):
    from hplot.stats import compute_layer_pvalues
    df = pd.read_csv(args.input)
    pvals = compute_layer_pvalues(
        df,
        prop=args.target,
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
        _cluster_mass_summary(df, pvals, args)


def _cluster_mass_summary(df, pvals, args):
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
                df_perm, prop=args.target, layer_col=args.layer,
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
    p.add_argument("--target",        required=True, help="Target proportion column.")
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
    p.add_argument("-o", "--output",  default=None,
                   help="Output CSV path for p-value table (stdout if omitted).")
    p.set_defaults(func=_cmd_test)


# ── gam ───────────────────────────────────────────────────────────────────

def _cmd_gam(args):
    from hplot.stats import gam_pooled_effect, gam_group_curves
    df = pd.read_csv(args.input)
    covariates = args.covariates or None
    effect, pval, n = gam_pooled_effect(
        long_df=df,
        target_col=args.target,
        layer_col=args.layer,
        group_col=args.group,
        at_layer=args.at_layer,
        groups=tuple(args.groups) if args.groups else None,
        covariate_cols=covariates,
        n_splines=args.n_splines,
    )
    cov_str = ", ".join(covariates) if covariates else "none"
    print(f"[hplot gam]  target={args.target}  group={args.group}  "
          f"at_layer={args.at_layer}")
    print(f"             covariates  : [{cov_str}]")
    print(f"             effect (high - low) = {effect:+.4f}")
    print(f"             p-value (group term) = {pval:.3e}   n = {n}")
    if args.curves_output:
        grid = np.arange(df[args.layer].min(), df[args.layer].max() + 1)
        curves = gam_group_curves(
            long_df=df,
            target_col=args.target,
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
    p.add_argument("--target",       required=True, help="Response column.")
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
    p.add_argument("--curves-output", dest="curves_output", default=None,
                   help="CSV path to save per-group GAM predictions + 95%% CI.")
    p.set_defaults(func=_cmd_gam)


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
            "",
            "Run  hplot <sub-command> --help  for full options.",
        ]),
    )
    sub = parser.add_subparsers(dest="command")
    _add_plot_parser(sub)
    _add_test_parser(sub)
    _add_gam_parser(sub)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
