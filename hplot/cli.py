"""Command line interface for hplot."""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from .runners import run_hplot_batch


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser shared by the CLI entry points."""

    parser = argparse.ArgumentParser(
        description="Generate H-Plots from spatial heterogeneity data.",
    )
    parser.add_argument(
        "input",
        help="Input CSV file path containing at least the value and layer columns.",
    )
    parser.add_argument(
        "--value-col",
        default="value",
        help="Column containing the value to average per layer.",
    )
    parser.add_argument(
        "--layer-col",
        default="layer",
        help="Column containing the discrete layer indices.",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Optional column that defines the curves to draw within each plot.",
    )
    parser.add_argument(
        "--case-col",
        default=None,
        help="Optional column identifying independent cases; one file per case is produced.",
    )
    parser.add_argument(
        "--distance-col",
        default=None,
        help="Optional column with the physical distance that corresponds to each layer.",
    )
    parser.add_argument(
        "--distance-unit",
        default=None,
        help="Label for the physical distance unit when --distance-col is provided.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence interval level used when shading the plot.",
    )
    parser.add_argument(
        "--hide-ci",
        action="store_true",
        help="Disable confidence interval shading in the output figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="hplots",
        help="Directory where the generated figures are written.",
    )
    parser.add_argument(
        "--file-prefix",
        default="hplot",
        help="Prefix for the output filenames.",
    )
    parser.add_argument(
        "--file-format",
        default="svg",
        choices=["svg", "pdf", "png"],
        help="Image format for the exported figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (dots per inch) when saving raster formats.",
    )
    parser.add_argument(
        "--display-base-type",
        default="tumor",
        help="Label for the base tissue/region shown on the x-axis title.",
    )
    parser.add_argument(
        "--display-target-type",
        default="immune cells",
        help="Label for the profiled cell type displayed on the y-axis title.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the console script."""

    parser = build_parser()
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)

    run_hplot_batch(
        df=df,
        value_col=args.value_col,
        layer_col=args.layer_col,
        group_col=args.group_col,
        case_col=args.case_col,
        distance_col=args.distance_col,
        distance_unit=args.distance_unit,
        ci=args.ci_level,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        ci_show=not args.hide_ci,
        file_format=args.file_format,
        dpi=args.dpi,
        display_base_type=args.display_base_type,
        display_target_type=args.display_target_type,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
