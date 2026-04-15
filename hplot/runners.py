import os
from .core import HPlot

def run_hplot_batch(
    df,
    targets="target_prop",
    layer="layer",
    group=None,
    distance=None,
    unit=None,
    ci=0.95,
    output="hplots",
    prefix="hplot",
    ci_show=True,
    format="svg",
    dpi=300,
):
    """
    根據群組自動產出多張 H-Plot 圖檔（每個 group 一張）
    """
    os.makedirs(output, exist_ok=True)

    if group and group in df.columns:
        groups = df[group].unique()
    else:
        groups = ["all"]

    for grp in groups:
        if grp == "all":
            sub_df = df
        else:
            sub_df = df[df[group] == grp]

        h = HPlot()
        h.fit(sub_df, targets=targets, layer=layer, group=group, distance=distance, unit=unit, ci=ci)
        h.plot(ci_show=ci_show)
        filename = os.path.join(output, f"{prefix}_{grp}.{format}")
        h.savefig(filename, dpi=dpi)