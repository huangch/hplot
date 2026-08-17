"""Hand-written CLI command table for the hplot MCP server.

Unlike the wsinsight / sptxinsight siblings (which auto-register tools
from a bundled or live ``describe`` JSON schema), hplot's CLI is plain
argparse with no ``describe`` command. The five sub-commands are small
and stable, so this module keeps a hand-written mirror of
``hplot/cli.py``: one entry per sub-command, one dict per parameter.

Parameter dict keys (a subset of the wsinsight schema convention):

* ``name``       — canonical snake_case name (argparse ``dest``).
* ``kind``       — ``string`` | ``int`` | ``float`` | ``bool``.
* ``required``   — True when the CLI requires the flag.
* ``default``    — CLI default (``None`` = no default / absent).
* ``flags``      — CLI flags; the adapter picks the first ``--`` one.
* ``is_flag``    — True for ``action="store_true"`` booleans.
* ``multiple``   — True for ``nargs`` options (values follow one flag).
* ``nargs``      — exact value count when the CLI uses a fixed ``nargs``.
* ``choices``    — allowed values (informational; not enforced here).
* ``help``       — the ``--help`` text for the flag.

Keep this table in sync with ``hplot/cli.py`` when the CLI changes.
"""

from __future__ import annotations

from typing import Any, Dict

# Shared screen arguments (``_add_screen_args`` in cli.py), used by both
# ``screen`` and ``loci --screen``.
_SCREEN_PARAMS: list[Dict[str, Any]] = [
    {"name": "sample", "kind": "string", "default": "sample",
     "flags": ["--sample"], "help": "Slide/sample id column."},
    {"name": "layer", "kind": "string", "default": "layer",
     "flags": ["--layer"], "help": "Signed layer index column."},
    {"name": "unit", "kind": "string", "default": "unit",
     "flags": ["--unit"], "help": "Feature column (gene / LR pair / cell type)."},
    {"name": "value", "kind": "string", "default": "value",
     "flags": ["--value"], "help": "Per-layer value column."},
    {"name": "distance", "kind": "string", "default": None,
     "flags": ["--distance"],
     "help": "Physical-distance (um) column; enables *_um outputs."},
    {"name": "grid", "kind": "int", "default": None, "multiple": True, "nargs": 2,
     "flags": ["--grid"],
     "help": "Analysis-window layer range LO HI (default: data min..max)."},
    {"name": "baseline", "kind": "string", "default": "window",
     "flags": ["--baseline"],
     "help": "Baseline region: window | far | core | 'a,b'."},
    {"name": "min_baseline_layers", "kind": "int", "default": 3,
     "flags": ["--min-baseline-layers"],
     "help": "Min baseline-region layers per slide (default 3)."},
    {"name": "band_mode", "kind": "string", "default": "dominant",
     "choices": ["dominant", "bidirectional"],
     "flags": ["--band-mode"],
     "help": "Winner-take-all (dominant) or per-direction bands."},
    {"name": "cluster_alpha", "kind": "float", "default": 0.05,
     "flags": ["--cluster-alpha"],
     "help": "Cluster-forming alpha (chi2 threshold; default 0.05)."},
    {"name": "min_w", "kind": "int", "default": 1,
     "flags": ["--min-w"],
     "help": "Minimum contiguous band width in layers (default 1)."},
    {"name": "min_per_group", "kind": "int", "default": 10,
     "flags": ["--min-per-group"],
     "help": "Minimum contributing slides per layer (default 10)."},
    {"name": "permutations", "kind": "int", "default": 1000,
     "flags": ["--permutations"],
     "help": "Layer-shuffle permutations (default 1000)."},
    {"name": "seed", "kind": "int", "default": 0,
     "flags": ["--seed"], "help": "Random seed (default 0)."},
    {"name": "progress", "kind": "bool", "is_flag": True,
     "flags": ["--progress"],
     "help": "Show a tqdm bar over permutations."},
]

COMMANDS: Dict[str, Dict[str, Any]] = {
    "plot": {
        "name": "plot",
        "help": "Draw H-Plot curves and save as SVG/PNG/PDF.",
        "params": [
            {"name": "input", "kind": "string", "required": True,
             "flags": ["-i", "--input"], "help": "Input CSV file."},
            {"name": "targets", "kind": "string", "default": ["target_prop"],
             "multiple": True, "flags": ["--targets"],
             "help": "Column name(s) for the target quantity."},
            {"name": "layer", "kind": "string", "default": "layer",
             "flags": ["--layer"], "help": "Layer index column."},
            {"name": "group", "kind": "string", "default": None,
             "flags": ["--group"], "help": "Group label column."},
            {"name": "distance", "kind": "string", "default": None,
             "flags": ["--distance"], "help": "Physical distance column."},
            {"name": "unit", "kind": "string", "default": None,
             "flags": ["-u", "--unit"],
             "help": "Distance unit label (e.g. um)."},
            {"name": "output", "kind": "string", "default": "hplots",
             "flags": ["-o", "--output"], "help": "Output directory."},
            {"name": "prefix", "kind": "string", "default": "hplot",
             "flags": ["-p", "--prefix"], "help": "Output filename prefix."},
            {"name": "format", "kind": "string", "default": "svg",
             "choices": ["svg", "pdf", "png"], "flags": ["-f", "--format"],
             "help": "Output image format."},
            {"name": "dpi", "kind": "int", "default": 300,
             "flags": ["--dpi"], "help": "DPI for PNG output."},
            {"name": "ci", "kind": "bool", "is_flag": True,
             "flags": ["--ci"],
             "help": "Show confidence interval bands."},
            {"name": "exclude_base", "kind": "bool", "is_flag": True,
             "flags": ["--exclude-base"],
             "help": "Exclude base cells from the denominator: "
                     "target_count / (all_count - base_count)."},
            {"name": "min_base_excluded_count", "kind": "int", "default": 1,
             "flags": ["--min-base-excluded-count"],
             "help": "Drop layers where all_count - base_count < this "
                     "(only with --exclude-base; default 1)."},
        ],
    },
    "test": {
        "name": "test",
        "help": "Per-layer Mann-Whitney test + optional cluster-mass permutation.",
        "params": [
            {"name": "input", "kind": "string", "required": True,
             "flags": ["-i", "--input"], "help": "Input CSV file."},
            {"name": "target", "kind": "string", "default": None,
             "flags": ["--target"],
             "help": "Target proportion column (required unless "
                     "--exclude-base is given)."},
            {"name": "layer", "kind": "string", "default": "layer",
             "flags": ["--layer"], "help": "Layer index column."},
            {"name": "group", "kind": "string", "required": True,
             "flags": ["--group"], "help": "Group label column."},
            {"name": "groups", "kind": "string", "default": None,
             "multiple": True, "nargs": 2, "flags": ["--groups"],
             "help": "Explicit group pair LOW HIGH (required when >2 "
                     "unique values)."},
            {"name": "distance", "kind": "string", "default": None,
             "flags": ["--distance"], "help": "Physical distance column."},
            {"name": "test", "kind": "string", "default": "mannwhitney",
             "choices": ["mannwhitney", "ttest", "welch"],
             "flags": ["--test"],
             "help": "Per-layer statistical test (default: mannwhitney)."},
            {"name": "correction", "kind": "string", "default": None,
             "choices": ["bonferroni", "fdr_bh"],
             "flags": ["--correction"],
             "help": "Multiple-testing correction across layers."},
            {"name": "min_n", "kind": "int", "default": 3,
             "flags": ["--min-n"],
             "help": "Min cases per group to test a layer (default 3)."},
            {"name": "permutations", "kind": "int", "default": 0,
             "flags": ["--permutations"],
             "help": "Label-permutations for cluster-mass test (0 = skip)."},
            {"name": "threshold", "kind": "float", "default": 0.05,
             "flags": ["--threshold"],
             "help": "Per-layer significance threshold for cluster-mass."},
            {"name": "seed", "kind": "int", "default": 42,
             "flags": ["--seed"], "help": "Random seed."},
            {"name": "exclude_base", "kind": "bool", "is_flag": True,
             "flags": ["--exclude-base"],
             "help": "Derive the target from counts as "
                     "target_count / (all_count - base_count) before testing."},
            {"name": "min_base_excluded_count", "kind": "int", "default": 1,
             "flags": ["--min-base-excluded-count"],
             "help": "Drop layers where all_count - base_count < this "
                     "(only with --exclude-base; default 1)."},
            {"name": "output", "kind": "string", "default": None,
             "flags": ["-o", "--output"],
             "help": "Output CSV path for p-value table (stdout if omitted)."},
        ],
    },
    "gam": {
        "name": "gam",
        "help": "Stage-2 GAM effect size with optional confounder adjustment.",
        "params": [
            {"name": "input", "kind": "string", "required": True,
             "flags": ["-i", "--input"], "help": "Input CSV file."},
            {"name": "target", "kind": "string", "default": None,
             "flags": ["--target"],
             "help": "Response column (required unless --exclude-base is "
                     "given)."},
            {"name": "layer", "kind": "string", "default": "layer",
             "flags": ["--layer"], "help": "Layer index column."},
            {"name": "group", "kind": "string", "required": True,
             "flags": ["--group"], "help": "Group label column."},
            {"name": "groups", "kind": "string", "default": None,
             "multiple": True, "nargs": 2, "flags": ["--groups"],
             "help": "Explicit (low, high) group pair."},
            {"name": "at_layer", "kind": "float", "required": True,
             "flags": ["--at-layer"],
             "help": "Layer at which to evaluate the group effect."},
            {"name": "covariates", "kind": "string", "default": None,
             "multiple": True, "flags": ["--covariates"],
             "help": "Columns to include as linear confounders."},
            {"name": "n_splines", "kind": "int", "default": 10,
             "flags": ["--n-splines"],
             "help": "Number of B-spline basis functions (default 10)."},
            {"name": "exclude_base", "kind": "bool", "is_flag": True,
             "flags": ["--exclude-base"],
             "help": "Derive the response from counts as "
                     "target_count / (all_count - base_count) before fitting."},
            {"name": "min_base_excluded_count", "kind": "int", "default": 1,
             "flags": ["--min-base-excluded-count"],
             "help": "Drop layers where all_count - base_count < this "
                     "(only with --exclude-base; default 1)."},
            {"name": "curves_output", "kind": "string", "default": None,
             "flags": ["--curves-output"],
             "help": "CSV path to save per-group GAM predictions + 95% CI."},
        ],
    },
    "screen": {
        "name": "screen",
        "help": "Multi-feature cluster-mass border-gradient screen -> ranking CSV.",
        "params": [
            {"name": "input", "kind": "string", "required": True,
             "flags": ["-i", "--input"],
             "help": "Long CSV: sample, layer, unit, value columns."},
            *_SCREEN_PARAMS,
            {"name": "output", "kind": "string", "default": "ranking.csv",
             "flags": ["-o", "--output"],
             "help": "Output ranking CSV (one row per banded feature)."},
            {"name": "wide_output", "kind": "string", "default": None,
             "flags": ["--wide-output"],
             "help": "Optional CSV for the per-feature wide table."},
        ],
    },
    "loci": {
        "name": "loci",
        "help": "Render an H-Loci Summary panel from a ranking CSV.",
        "params": [
            {"name": "input", "kind": "string", "required": True,
             "flags": ["-i", "--input"],
             "help": "Ranking CSV (or raw long CSV when --screen is set)."},
            {"name": "output", "kind": "string", "default": "hloci.svg",
             "flags": ["-o", "--output"],
             "help": "Output figure path (.svg/.pdf/.png)."},
            {"name": "kind", "kind": "string", "default": "bands",
             "choices": ["bands", "summary", "bidirectional"],
             "flags": ["--kind"],
             "help": "Panel style: bands (default, canonical band view) | "
                     "bidirectional | summary (legacy strip+triangle)."},
            {"name": "sort", "kind": "string", "default": "outer_to_inner",
             "choices": ["outer_to_inner", "inner_to_outer", "none"],
             "flags": ["--sort"],
             "help": "Row ordering by band centre (default outer_to_inner)."},
            {"name": "top_n", "kind": "int", "default": None,
             "flags": ["--top-n"],
             "help": "Keep the top-N rows by cluster mass before drawing."},
            {"name": "width", "kind": "float", "default": 6.4,
             "flags": ["--width"], "help": "Figure width (in)."},
            {"name": "dpi", "kind": "int", "default": 300,
             "flags": ["--dpi"], "help": "Raster DPI (default 300)."},
            {"name": "title", "kind": "string", "default": None,
             "flags": ["--title"], "help": "Panel title."},
            {"name": "label_col", "kind": "string", "default": "gene",
             "flags": ["--label-col"],
             "help": "Ranking-table column holding feature labels."},
            {"name": "lo_col", "kind": "string", "default": "band_start_layer",
             "flags": ["--lo-col"],
             "help": "Band start-layer column."},
            {"name": "hi_col", "kind": "string", "default": "band_end_layer",
             "flags": ["--hi-col"], "help": "Band end-layer column."},
            {"name": "dir_col", "kind": "string", "default": "direction",
             "flags": ["--dir-col"], "help": "Direction column."},
            {"name": "peak_col", "kind": "string", "default": "peak_layer",
             "flags": ["--peak-col"], "help": "Peak-layer column."},
            {"name": "mass_col", "kind": "string", "default": "cluster_mass",
             "flags": ["--mass-col"], "help": "Cluster-mass column."},
            {"name": "fdr_col", "kind": "string", "default": "fdr",
             "flags": ["--fdr-col"], "help": "FDR column."},
            {"name": "fdr_max", "kind": "float", "default": None,
             "flags": ["--fdr-max"],
             "help": "Drop rows with FDR above this before drawing."},
            {"name": "screen", "kind": "bool", "is_flag": True,
             "flags": ["--screen"],
             "help": "Run 'hplot screen' first (input is a raw long CSV)."},
            *_SCREEN_PARAMS,
        ],
    },
}

#: Every sub-command is stable (hplot has no experimental split).
STABLE = frozenset(COMMANDS)

#: Long-running sub-commands -> job_id + polling. ``test`` runs label
#: permutations, ``screen`` runs 1000 layer-shuffle permutations by
#: default, and ``loci`` can chain the full screen via --screen.
LONG_RUNNING = frozenset({"test", "screen", "loci"})


def discover_commands() -> Dict[str, Dict[str, Any]]:
    """Return the command table (all sub-commands are stable)."""
    return {name: dict(cmd) for name, cmd in COMMANDS.items()}


def is_long_running(name: str) -> bool:
    """True when the sub-command should run as a background job."""
    return name in LONG_RUNNING


__all__ = [
    "COMMANDS",
    "STABLE",
    "LONG_RUNNING",
    "discover_commands",
    "is_long_running",
]
