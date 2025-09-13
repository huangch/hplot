import os
from .core import HPlot

def run_hplot_batch(
    df,
    value_col="value",
    layer_col="layer",
    region_col="region_id",
    group_col=None,
    distance_col=None,
    distance_unit=None,
    label_col=None,
    ci=None,
    output_dir="hplots",
    file_prefix="hplot",
    ci_show=True,
    file_format="svg",
    dpi=300,
):
    """
    根據群組自動產出多張 H-Plot 圖檔（每個 group 一張）
    """
    os.makedirs(output_dir, exist_ok=True)

    if label_col and label_col in df.columns:
        groups = df[label_col].unique()
    else:
        groups = ["all"]

    for group in groups:
        if group == "all":
            sub_df = df
        else:
            sub_df = df[df[label_col] == group]

        h = HPlot()
        h.fit(sub_df, value_col=value_col, layer_col=layer_col, region_col=region_col, group_col=group_col, distance_col=distance_col, distance_unit=distance_unit, ci=ci)
        h.plot(ci_show=ci_show)
        filename = os.path.join(output_dir, f"{file_prefix}_{group}.{file_format}")
        h.savefig(filename, dpi=dpi)