"""hplot command-line interface (click-based).

Sub-commands
------------
hplot plot   — draw H-Plot curves from a CSV (batch-safe)
hplot test   — per-layer Mann-Whitney / cluster-mass permutation test
hplot gam    — Stage-2 GAM effect size with optional confounder adjustment
hplot screen — multi-feature cluster-mass border-gradient screen -> ranking CSV
hplot loci   — render an H-Loci Summary panel from a ranking CSV
hplot schema — machine-readable JSON schema of every hplot sub-command

Usage examples
--------------
::

    hplot plot  -i data.csv --target immune_fraction --group hpv_status -o out/

    hplot test  -i data.csv --target immune_fraction --group hpv_status \\
                --permutations 999 -o out/pvalues.csv

    hplot gam   -i data.csv --target immune_fraction --group hpv_status \\
                --at-layer 0 --covariates AGE late_stage is_female

"""

from __future__ import annotations

import json as _json
import os
import sys

import click

import numpy as np
import pandas as pd


# Re-exported name used by tools/mcp/server.py for click error formatting
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def _out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ── shared screen options (used by 'screen' and 'loci --screen') ──────────────
#
# NOTE: click does not allow a single decorator to inject option flags into an
# existing command (only @click.option chained directly works). We define two
# parallel lists of click.option decorators and apply both to 'screen' and 'loci'.
# This duplicates the option declarations, but is the canonical click pattern.

def _screen_option_specs():
    """Helper for tests only: returns the ordered list of (name, dest, optobj)
    triples. Callers should use the `_SCREEN_OPTIONS` decorator chain directly."""
    # Not used at runtime; kept for clarity.
    pass


# ── screen / loci helpers (unchanged logic; click passes args as kwargs) ─────


def _pivot_slides(df, sample_col, layer_col, unit_col, value_col):
    """Long CSV -> (values, layers, unit_names) for ``deviation_tensor``."""
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


class _SimpleNamespace:
    """Tiny stand-in for ``argparse.Namespace`` so screen results pass into
    ``_run_screen_from_csv`` unchanged."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _run_screen_from_csv(args):
    """Shared screen driver used by ``hplot screen`` and ``hplot loci --screen``."""
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


# ── loci: post-screen layer_um recovery (unchanged logic) ─────────────────────


def _layer_um_from_ranking(df):
    """Reconstruct a ``{layer L: physical distance µm}`` map from a ranking
    table using its (layer, µm) column pairs."""
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


# ── subcommand callback implementations ────────────────────────────────────


def _cmd_plot(input, targets, layer, group, distance, unit, output, prefix,
              format, dpi, ci, exclude_base, min_base_excluded_count):
    from hplot.runners import run_hplot_batch
    df = pd.read_csv(input)
    run_hplot_batch(
        df=df,
        targets=list(targets) if targets else ["target_prop"],
        layer=layer,
        group=group,
        distance=distance,
        unit=unit,
        output=_out_dir(output),
        prefix=prefix,
        ci_show=ci,
        format=format,
        dpi=dpi,
        exclude_base=exclude_base,
        min_base_excluded_count=min_base_excluded_count,
    )
    click.echo(f"[hplot plot]  figures written to {output}/")


def _cmd_test(input, target, layer, group, groups, distance, test,
              correction, min_n, permutations, threshold, seed,
              exclude_base, min_base_excluded_count, output):
    from hplot.stats import compute_layer_pvalues
    df = pd.read_csv(input)
    eff_target = target
    if exclude_base:
        from hplot.runners import add_base_excluded_proportion
        df, eff_target = add_base_excluded_proportion(
            df, min_base_excluded_count=min_base_excluded_count)
    elif eff_target is None:
        raise click.UsageError(
            "hplot test: --target is required unless --exclude-base is given.")
    pvals = compute_layer_pvalues(
        df,
        prop=eff_target,
        layer_col=layer,
        group_col=group,
        groups=tuple(groups) if groups else None,
        test=test,
        distance_col=distance,
        min_n=min_n,
        correction=correction,
    )
    if output:
        pvals.to_csv(output, index=False)
        click.echo(f"[hplot test]  p-value table written to {output}")
    else:
        click.echo(pvals.to_string(index=False))

    if permutations > 0:
        col = "p_adj" if correction else "p_value"
        sig = pvals[pvals[col] < threshold]
        if sig.empty:
            click.echo(f"[hplot test]  No layers significant at p < {threshold}; "
                       "cluster-mass not computed.")
            return
        obs_mass = (threshold - sig[col]).clip(lower=0).sum()
        rng = np.random.default_rng(seed)
        null_masses = []
        for _ in range(permutations):
            df_perm = df.copy()
            g_vals = df_perm[group].to_numpy()
            rng.shuffle(g_vals)
            df_perm[group] = g_vals
            try:
                pv_perm = compute_layer_pvalues(
                    df_perm, prop=eff_target, layer_col=layer,
                    group_col=group,
                    groups=tuple(groups) if groups else None,
                    test=test, distance_col=distance,
                    min_n=min_n, correction=correction,
                )
                sig_p = pv_perm[pv_perm[col] < threshold]
                null_masses.append((threshold - sig_p[col]).clip(lower=0).sum())
            except Exception:
                null_masses.append(0.0)
        perm_p = float((np.array(null_masses) >= obs_mass).mean())
        click.echo(f"[hplot test]  Cluster-mass: obs={obs_mass:.4f}  "
                   f"perm-p={perm_p:.4f}  (n_perm={permutations},"
                   f" threshold={threshold})")


def _cmd_gam(input, target, layer, group, groups, at_layer, covariates,
             n_splines, exclude_base, min_base_excluded_count, curves_output):
    from hplot.stats import gam_pooled_effect, gam_group_curves
    df = pd.read_csv(input)
    eff_target = target
    if exclude_base:
        from hplot.runners import add_base_excluded_proportion
        df, eff_target = add_base_excluded_proportion(
            df, min_base_excluded_count=min_base_excluded_count)
    elif eff_target is None:
        raise click.UsageError(
            "hplot gam: --target is required unless --exclude-base is given.")
    cov = covariates or None
    effect, pval, n = gam_pooled_effect(
        long_df=df,
        target_col=eff_target,
        layer_col=layer,
        group_col=group,
        at_layer=at_layer,
        groups=tuple(groups) if groups else None,
        covariate_cols=cov,
        n_splines=n_splines,
    )
    cov_str = ", ".join(covariates) if covariates else "none"
    click.echo(f"[hplot gam]  target={eff_target}  group={group}  "
               f"at_layer={at_layer}")
    click.echo(f"             covariates  : [{cov_str}]")
    click.echo(f"             effect (high - low) = {effect:+.4f}")
    click.echo(f"             p-value (group term) = {pval:.3e}   n = {n}")
    if curves_output:
        grid = np.arange(df[layer].min(), df[layer].max() + 1)
        curves = gam_group_curves(
            long_df=df,
            target_col=eff_target,
            layer_col=layer,
            group_col=group,
            grid=grid,
            groups=tuple(groups) if groups else None,
            n_splines=n_splines,
        )
        rows = []
        for grp, (pred, ci) in curves.items():
            for i, lyr in enumerate(grid):
                rows.append({"group": grp, "layer": lyr,
                             "gam_pred": pred[i],
                             "ci_lower": ci[i, 0], "ci_upper": ci[i, 1]})
        pd.DataFrame(rows).to_csv(curves_output, index=False)
        click.echo(f"[hplot gam]  curve predictions written to {curves_output}")


def _cmd_screen(input, output, wide_output, **screen_kwargs):
    args = _SimpleNamespace(input=input, output=output, **screen_kwargs)
    long_df, wide_df, _ = _run_screen_from_csv(args)
    long_df.to_csv(output, index=False)
    n_band = len(long_df)
    click.echo(f"[hplot screen]  ranking table ({n_band} banded rows) -> {output}")
    if wide_output is not None:
        wide_df.to_csv(wide_output, index=False)
        click.echo(f"[hplot screen]  wide table -> {wide_output}")


# Screen option keys, used by loci's --screen to forward to _run_screen_from_csv
_SCREEN_KW_KEYS = frozenset({
    "sample", "layer", "unit", "value", "distance", "grid",
    "baseline", "min_baseline_layers", "band_mode", "cluster_alpha",
    "min_w", "min_per_group", "permutations", "seed", "progress",
})


def _cmd_loci(**kw):
    """loci dispatcher.

    Click passes a single ``**kw`` because the option set is the union of
    loci-specific + screen-specific flags (when --screen is set).
    """
    screen_flag = kw.pop("screen", False)
    if screen_flag:
        screen_opts = {k: kw.pop(k) for k in _SCREEN_KW_KEYS if k in kw}
        _cmd_loci(screen=True, **screen_opts, **kw)
    else:
        # drop screen kwargs that aren't relevant when not chaining a screen
        for k in _SCREEN_KW_KEYS:
            kw.pop(k, None)
        _cmd_loci_inner(**kw)


def _cmd_loci_inner(input, output, kind, sort, top_n, width, dpi, title,
                    label_col, lo_col, hi_col, dir_col, peak_col, mass_col,
                    fdr_col, fdr_max, screen, **_screen_ignored):
    # When called from _cmd_loci with --screen, screen and *_screen_ignored
    # carry the screen arguments we just consumed. They're not used here.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hplot

    if screen:
        # Re-run screen with the original args (which the loci dispatcher kept),
        # then plot from that result.
        loci_kwargs = {k: v for k, v in {
            "input": input, "output": output, "kind": kind, "sort": sort,
            "top_n": top_n, "width": width, "dpi": dpi, "title": title,
            "label_col": label_col, "lo_col": lo_col, "hi_col": hi_col,
            "dir_col": dir_col, "peak_col": peak_col, "mass_col": mass_col,
            "fdr_col": fdr_col, "fdr_max": fdr_max,
        }.items()}
        # Caller will have already populated _screen_ignored with _SCREEN_KW_KEYS
        screen_ns = _SimpleNamespace(**_screen_ignored)
        long_df, wide_df, layer_um = _run_screen_from_csv(screen_ns)
        df = wide_df if kind == "bidirectional" else long_df
    else:
        df = pd.read_csv(input)
        layer_um = _layer_um_from_ranking(df)

    if fdr_max is not None and fdr_col in df.columns:
        df = df[df[fdr_col] <= fdr_max]
    if top_n and mass_col in df.columns:
        df = df.sort_values(mass_col, ascending=False).head(top_n)
    if df.empty:
        raise click.UsageError("hplot loci: no rows to plot after filtering.")

    sort = None if sort == "none" else sort
    n = len(df)
    fig_h = float(np.clip(0.45 * n + 2.4, 4.0, 24.0))
    fig, ax = plt.subplots(figsize=(width, fig_h))

    if kind == "bands":
        hplot.plot_hloci_bands(
            df[lo_col], df[hi_col], df[dir_col],
            peak=df[peak_col] if peak_col in df.columns else None,
            mass=df[mass_col] if mass_col in df.columns else None,
            labels=df[label_col], sort=sort, ax=ax,
            xlabel="border layer L", title=title)
    elif kind == "summary":
        hplot.plot_hloci_strip(
            df[peak_col], df[dir_col],
            weights=df[mass_col] if mass_col in df.columns else None,
            labels=df[label_col], ax=ax,
            xlabel="border layer L", title=title)
    else:
        hplot.plot_hloci_bands_bidir(
            df[label_col],
            df["elevated_start"], df["elevated_end"],
            df["depressed_start"], df["depressed_end"],
            elev_center=df.get("elevated_center"),
            depr_center=df.get("depressed_center"),
            elev_mass=df.get("elevated_mass"),
            depr_mass=df.get("depressed_mass"),
            sort_by=None if sort is None else "dominant_center",
            ax=ax, title=title)

    if layer_um is not None:
        hplot.add_border_distance_axis(ax, layer_um)

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    click.echo(f"[hplot loci]  {kind} panel ({n} rows) -> {output}")


def _cmd_schema(output, commands_only):
    """Emit machine-readable JSON of hplot sub-commands.

    ``commands_only`` emits a flat list of registered subcommand names — used by
    the ``hplot.sh`` wrapper for cheap subcommand-boundary discovery.
    """
    from hplot.mcp.schema import COMMANDS
    if commands_only:
        payload = _json.dumps(
            {"schema_version": 1, "commands": sorted(COMMANDS.keys())},
            indent=2, sort_keys=True,
        )
    else:
        payload = _json.dumps(
            {"schema_version": 1, "commands": COMMANDS},
            indent=2, sort_keys=True, default=str,
        )
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    else:
        click.echo(payload)


# ── click subcommand definitions ──────────────────────────────────────────


@click.group(context_settings=CONTEXT_SETTINGS, help="hplot — H-Plot border-profiling toolkit.")
@click.pass_context
def cli(ctx):
    """hplot: graph-geodesic spatial profiling at tissue boundaries."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@cli.command("plot", help="Draw H-Plot curves and save as SVG/PNG/PDF.")
@click.option("-i", "--input", required=True, help="Input CSV file.")
@click.option("--targets", multiple=True, default=("target_prop",),
              help="Column name(s) for the target quantity.")
@click.option("--layer", default="layer",  help="Layer index column.")
@click.option("--group", default=None,      help="Group label column.")
@click.option("--distance", default=None,   help="Physical distance column.")
@click.option("-u", "--unit", default=None,  help="Distance unit label (e.g. um).")
@click.option("-o", "--output", default="hplots", help="Output directory.")
@click.option("-p", "--prefix", default="hplot", help="Output filename prefix.")
@click.option("-f", "--format", default="svg",
              type=click.Choice(["svg", "pdf", "png"]),
              help="Output image format.")
@click.option("--dpi", type=int, default=300, help="DPI for PNG output.")
@click.option("--ci", is_flag=True, help="Show confidence interval bands.")
@click.option("--exclude-base", is_flag=True,
              help="Exclude base cells from the denominator: "
                   "target_count / (all_count - base_count).")
@click.option("--min-base-excluded-count", "min_base_excluded_count", type=int, default=1,
              help="Drop layers where all_count - base_count < this "
                   "(only with --exclude-base; default 1).")
def plot(**kw):
    _cmd_plot(**kw)


@cli.command("test",
             help="Per-layer Mann-Whitney test + optional cluster-mass permutation.")
@click.option("-i", "--input", required=True, help="Input CSV file.")
@click.option("--target", default=None,
              help="Target proportion column (required unless --exclude-base).")
@click.option("--layer", default="layer", help="Layer index column.")
@click.option("--group", required=True, help="Group label column.")
@click.option("--groups", nargs=2, default=None, metavar="LOW HIGH",
              help="Explicit group pair (required when >2 unique values).")
@click.option("--distance", default=None, help="Physical distance column.")
@click.option("--test", default="mannwhitney",
              type=click.Choice(["mannwhitney", "ttest", "welch"]),
              help="Per-layer statistical test (default: mannwhitney).")
@click.option("--correction", default=None,
              type=click.Choice(["bonferroni", "fdr_bh"]),
              help="Multiple-testing correction across layers.")
@click.option("--min-n", "min_n", type=int, default=3,
              help="Min cases per group to test a layer (default 3).")
@click.option("--permutations", type=int, default=0,
              help="Label-permutations for cluster-mass test (0 = skip).")
@click.option("--threshold", type=float, default=0.05,
              help="Per-layer significance threshold for cluster-mass.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--exclude-base", is_flag=True,
              help="Derive the target from counts as "
                   "target_count / (all_count - base_count) before testing.")
@click.option("--min-base-excluded-count", "min_base_excluded_count", type=int, default=1,
              help="Drop layers where all_count - base_count < this "
                   "(only with --exclude-base; default 1).")
@click.option("-o", "--output", default=None,
              help="Output CSV path for p-value table (stdout if omitted).")
def test(**kw):
    _cmd_test(**kw)


@cli.command("gam",
             help="Stage-2 GAM effect size with optional confounder adjustment.")
@click.option("-i", "--input", required=True, help="Input CSV file.")
@click.option("--target", default=None,
              help="Response column (required unless --exclude-base).")
@click.option("--layer", default="layer", help="Layer index column.")
@click.option("--group", required=True, help="Group label column.")
@click.option("--groups", nargs=2, default=None, metavar="LOW HIGH",
              help="Explicit (low, high) group pair.")
@click.option("--at-layer", "at_layer", type=float, required=True,
              help="Layer at which to evaluate the group effect.")
@click.option("--covariates", multiple=True, default=None, metavar="COL",
              help="Columns to include as linear confounders.")
@click.option("--n-splines", "n_splines", type=int, default=10,
              help="Number of B-spline basis functions (default 10).")
@click.option("--exclude-base", is_flag=True,
              help="Derive the response from counts as "
                   "target_count / (all_count - base_count) before fitting.")
@click.option("--min-base-excluded-count", "min_base_excluded_count", type=int, default=1,
              help="Drop layers where all_count - base_count < this "
                   "(only with --exclude-base; default 1).")
@click.option("--curves-output", "curves_output", default=None,
              help="CSV path to save per-group GAM predictions + 95% CI.")
def gam(**kw):
    _cmd_gam(**kw)


@cli.command("screen",
             help="Multi-feature cluster-mass border-gradient screen -> ranking CSV.")
@click.option("-i", "--input", required=True,
              help="Long CSV: sample, layer, unit, value columns.")
# Inline shared screen options (click limitation: cannot inject via decorator)
@click.option("--sample", default="sample", help="Slide/sample id column.")
@click.option("--layer",  default="layer",  help="Signed layer index column.")
@click.option("--unit",   default="unit",
              help="Feature column (gene / LR pair / cell type).")
@click.option("--value",  default="value",  help="Per-layer value column.")
@click.option("--distance", default=None,
              help="Physical-distance (µm) column; enables *_um outputs.")
@click.option("--grid", nargs=2, type=int, default=None, metavar="LO HI",
              help="Analysis-window layer range (default: data min..max).")
@click.option("--baseline", default="window",
              help="Baseline region: window | far | core | 'a,b'.")
@click.option("--min-baseline-layers", "min_baseline_layers", type=int, default=3,
              help="Min baseline-region layers per slide (default 3).")
@click.option("--band-mode", "band_mode", default="dominant",
              type=click.Choice(["dominant", "bidirectional"]),
              help="Winner-take-all (dominant) or per-direction bands.")
@click.option("--cluster-alpha", "cluster_alpha", type=float, default=0.05,
              help="Cluster-forming alpha (chi2 threshold; default 0.05).")
@click.option("--min-w", "min_w", type=int, default=1,
              help="Minimum contiguous band width in layers (default 1).")
@click.option("--min-per-group", "min_per_group", type=int, default=10,
              help="Minimum contributing slides per layer (default 10).")
@click.option("--permutations", type=int, default=1000,
              help="Layer-shuffle permutations (default 1000).")
@click.option("--seed", type=int, default=0, help="Random seed (default 0).")
@click.option("--progress", is_flag=True,
              help="Show a tqdm bar over permutations.")
@click.option("-o", "--output", default="ranking.csv",
              help="Output ranking CSV (one row per banded feature).")
@click.option("--wide-output", "wide_output", default=None,
              help="Optional CSV for the per-feature wide table.")
def screen(**kw):
    """screen dispatcher."""
    input = kw.pop("input")
    output = kw.pop("output")
    wide_output = kw.pop("wide_output")
    _cmd_screen(input=input, output=output, wide_output=wide_output, **kw)


@cli.command("loci", help="Render an H-Loci Summary panel from a ranking CSV.")
@click.option("-i", "--input", required=True,
              help="Ranking CSV (or raw long CSV when --screen is set).")
@click.option("-o", "--output", default="hloci.svg",
              help="Output figure path (.svg/.pdf/.png).")
@click.option("--kind", default="bands",
              type=click.Choice(["bands", "summary", "bidirectional"]),
              help="Panel style: bands (default) | bidirectional | summary.")
@click.option("--sort", default="outer_to_inner",
              type=click.Choice(["outer_to_inner", "inner_to_outer", "none"]),
              help="Row ordering by band centre.")
@click.option("--top-n", "top_n", type=int, default=None,
              help="Keep the top-N rows by cluster mass before drawing.")
@click.option("--width", type=float, default=6.4, help="Figure width (in).")
@click.option("--dpi", type=int, default=300, help="Raster DPI (default 300).")
@click.option("--title", default=None, help="Panel title.")
@click.option("--label-col", "label_col", default="gene")
@click.option("--lo-col",   "lo_col",   default="band_start_layer")
@click.option("--hi-col",   "hi_col",   default="band_end_layer")
@click.option("--dir-col",  "dir_col",  default="direction")
@click.option("--peak-col", "peak_col", default="peak_layer")
@click.option("--mass-col", "mass_col", default="cluster_mass")
@click.option("--fdr-col",  "fdr_col",  default="fdr")
@click.option("--fdr-max",  "fdr_max",  type=float, default=None,
              help="Drop rows with FDR above this before drawing.")
@click.option("--screen", is_flag=True,
              help="Run 'hplot screen' first (input is a raw long CSV).")
# Inline shared screen options (click limitation: cannot inject via decorator)
@click.option("--sample", default="sample", help="Slide/sample id column.")
@click.option("--layer",  default="layer",  help="Signed layer index column.")
@click.option("--unit",   default="unit",
              help="Feature column (gene / LR pair / cell type).")
@click.option("--value",  default="value",  help="Per-layer value column.")
@click.option("--distance", default=None,
              help="Physical-distance (µm) column; enables *_um outputs.")
@click.option("--grid", nargs=2, type=int, default=None, metavar="LO HI",
              help="Analysis-window layer range (default: data min..max).")
@click.option("--baseline", default="window",
              help="Baseline region: window | far | core | 'a,b'.")
@click.option("--min-baseline-layers", "min_baseline_layers", type=int, default=3,
              help="Min baseline-region layers per slide (default 3).")
@click.option("--band-mode", "band_mode", default="dominant",
              type=click.Choice(["dominant", "bidirectional"]),
              help="Winner-take-all (dominant) or per-direction bands.")
@click.option("--cluster-alpha", "cluster_alpha", type=float, default=0.05,
              help="Cluster-forming alpha (chi2 threshold; default 0.05).")
@click.option("--min-w", "min_w", type=int, default=1,
              help="Minimum contiguous band width in layers (default 1).")
@click.option("--min-per-group", "min_per_group", type=int, default=10,
              help="Minimum contributing slides per layer (default 10).")
@click.option("--permutations", type=int, default=1000,
              help="Layer-shuffle permutations (default 1000).")
@click.option("--seed", type=int, default=0, help="Random seed (default 0).")
@click.option("--progress", is_flag=True,
              help="Show a tqdm bar over permutations.")
def loci(**kw):
    """loci dispatcher. Pop screen options before forwarding."""
    _cmd_loci(**kw)


@cli.command("schema",
             help="Emit a machine-readable JSON schema of every hplot sub-command.")
@click.option("--output", default=None,
              help="Write the schema JSON to this file instead of stdout.")
@click.option("--commands-only", is_flag=True, default=False,
              help="Emit only the names of the registered subcommands as a flat "
                   "JSON list, instead of the full per-command descriptor. "
                   "Cheaper for callers (e.g. hplot.sh) that only need to know "
                   "what subcommands exist.")
def schema(**kw):
    _cmd_schema(**kw)


# Console-script entry-point (pyproject.toml: project.scripts → "hplot.cli:main").
# The CLI's primary symbol is the click `cli` group; alias it to `main` so the
# auto-generated launcher (`from hplot.cli import main`) resolves correctly.
main = cli


if __name__ == "__main__":
    cli()
