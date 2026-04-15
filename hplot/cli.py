import argparse
import pandas as pd
from hplot.runners import run_hplot_batch


def main():
    parser = argparse.ArgumentParser(description="Generate H-Plot from spatial heterogeneity data.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("--targets", nargs="+", default=["target_prop"], help="One or more column names for target cell proportions (each becomes a separate line on the plot)")
    parser.add_argument("--layer", default="layer", help="Column name for the layer distance")
    parser.add_argument("--group", default=None, help="Column name for group (e.g. subtype within region)")
    parser.add_argument("--distance", default=None, help="Column name for distance (e.g. actual Euclidean distance from border rater than layer index)")
    parser.add_argument("-u", "--unit", default=None, help="distance unit, default: None")
    parser.add_argument("-o", "--output", default="hplots", help="Directory to save plots")
    parser.add_argument("-p", "--prefix", default="hplot", help="Prefix for output file names")
    parser.add_argument("-f", "--format", default="svg", choices=["svg", "pdf", "png"], help="Output image format")
    parser.add_argument("--ci", action="store_true", help="Whether to show confidence intervals")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    run_hplot_batch(
        df=df,
        targets=args.targets,
        layer=args.layer,
        group=args.group,
        distance=args.distance,
        unit=args.unit,
        output=args.output,
        prefix=args.prefix,
        ci_show=args.ci,
        format=args.format,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()