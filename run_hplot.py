import argparse
import pandas as pd
from hplot.runners import run_hplot_batch

def main():
    parser = argparse.ArgumentParser(description="Generate H-Plot from spatial heterogeneity data.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--value_col", default="value", help="Column name for the value (e.g. proportion)")
    parser.add_argument("--layer_col", default="layer", help="Column name for the layer distance")
    parser.add_argument("--case_col", default="case_id", help="Column name for tumor case ID")
    parser.add_argument("--group_col", default=None, help="Column name for group (e.g. subtype within region)")
    parser.add_argument("--distance_col", default=None, help="Column name for distance (e.g. actual Euclidean distance from border rater than layer index)")
    parser.add_argument("--distance_unit", default=None, help="distance unit, default: None")
    # parser.add_argument("--label_col", default=None, help="Column name for label (e.g. tumor type)")
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
        case_col=args.case_col,
        group_col=args.group_col,
        # label_col=args.label_col,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        ci_show=args.ci,
        file_format=args.file_format,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()