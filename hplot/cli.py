import json
import click
import argparse
import pandas as pd
from hplot.runners import run_hplot_batch
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(package_name="hplot")
def main():
    """H-plot CLI: spatial heterogeneity analysis & plots."""

@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--out", "-o", type=click.Path(dir_okay=False, path_type=str),
              default="hplot.png", show_default=True, help="Output plot file.")
@click.option("--layer-col", default="layer", show_default=True,
              help="Column indicating region layers.")
@click.option("--target-col", default="cell_type", show_default=True,
              help="Column with the cell type/category to profile.")
@click.option("--target", required=True,
              help="Which cell type/category ratio to plot (e.g., 'lymphocyte').")
# def plot(input_path, out, layer_col, target_col, target):
#     """Generate an H-plot from a prepared CSV/Parquet."""
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
#     if layer_col not in df or target_col not in df:
#         raise click.ClickException(f"Missing required columns: {layer_col}, {target_col}")
#     # toy example: compute ratio per layer
#     total = df.groupby(layer_col).size().rename("n_total")
#     hits = df[df[target_col] == target].groupby(layer_col).size().rename("n_target")
#     res = (
#         pd.concat([total, hits], axis=1)
#         .fillna(0.0)
#         .assign(ratio=lambda x: x["n_target"] / x["n_total"].where(x["n_total"] > 0, 1))
#         .reset_index()
#         .sort_values(layer_col)
#     )
#     plt.figure()
#     plt.plot(res[layer_col], res["ratio"], marker="o")
#     plt.xlabel(f"{layer_col} (distance layers)")
#     plt.ylabel(f"Ratio of {target}")
#     plt.title(f"H-plot: {target}")
#     plt.tight_layout()
#     plt.savefig(out)
#     click.echo(f"[hplot] saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate H-Plot from spatial heterogeneity data.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--value_col", default="value", help="Column name for the value (e.g. proportion)")
    parser.add_argument("--layer_col", default="layer", help="Column name for the layer distance")
    parser.add_argument("--group_col", default=None, help="Column name for group (e.g. subtype within region)")
    parser.add_argument("--distance_col", default=None, help="Column name for distance (e.g. actual Euclidean distance from border rater than layer index)")
    parser.add_argument("--distance_unit", default=None, help="distance unit, default: None")
    parser.add_argument("--output_dir", default="hplots", help="Directory to save plots")
    parser.add_argument("--file_prefix", default="hplot", help="Prefix for output file names")
    parser.add_argument("--file_format", default="svg", choices=["svg", "pdf", "png"], help="Output image format")
    parser.add_argument("--ci", action="store_true", help="Whether to show confidence intervals")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    run_hplot_batch(
        df=df,
        value_col=args.value_col,
        layer_col=args.layer_col,
        group_col=args.group_col,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        ci_show=args.ci,
        file_format=args.file_format,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()