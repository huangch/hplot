# hplot

**H-Plot: A spatial heterogeneity visualization for tissue-based distance layers**

`hplot` is a Python package for visualizing the spatial distribution of cell-type proportions across concentric distance layers measured from a tissue boundary (e.g., a tumor border). Inspired by Kaplan-Meier survival curves, the H-Plot replaces time with spatial layer index on the x-axis, making it easy to see how cell composition changes as you move inward or outward across a tissue region.

Both a **target** proportion (e.g., immune cells) and an optional **base** proportion (e.g., total epithelial cells) can be plotted together, with per-layer confidence intervals derived from across-case variability.

---

## Installation

```bash
pip install -e .
```

**Dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`

---

## Input Data Format

The input is a CSV file where each row represents one tissue region (case) at one spatial layer.

| case_id | layer | target_prop | base_prop | subtype | distance |
|---------|-------|-------------|-----------|---------|----------|
| C1      | -2    | 0.05        | 0.40      | hot     | 210.3    |
| C1      | -1    | 0.08        | 0.38      | hot     | 105.1    |
| C1      |  0    | 0.15        | 0.35      | hot     | 0.0      |
| C1      |  1    | 0.20        | 0.30      | hot     | 98.7     |
| C2      | -1    | 0.03        | 0.50      | cold    | 112.0    |
| ...     | ...   | ...         | ...       | ...     | ...      |

### Column descriptions

| Column | Required | Description |
|--------|----------|-------------|
| `layer` | Yes | Integer layer index. `0` = tissue boundary; negative = outside; positive = inside. |
| `target_prop` | Yes | Proportion of the target cell type (e.g., immune cells) in that layer for that case. |
| `base_prop` | No | Proportion of a base cell type (e.g., epithelial cells) to overlay for reference. |
| `case_id` / `group_col` | No | Groups rows into separate H-Plot lines (e.g., tumor subtype). |
| `distance` | No | Mean physical distance (in µm or other unit) corresponding to each layer index. Used for secondary x-axis tick labels. |

---

## Python API

### Basic usage

```python
import pandas as pd
from hplot.core import HPlot

df = pd.read_csv("input.csv")

hplot = HPlot()
hplot.fit(
    df,
    target_prop_col="target_prop",
    layer_col="layer",
    group_col="subtype",        # draw one line per subtype
    base_prop_col="base_prop",  # optional: overlay base cell proportion
    distance_col="distance",    # optional: physical distances for tick labels
    distance_unit="µm",         # optional: unit shown in x-axis label
    ci=0.95,                    # confidence interval level (default 0.95)
)
hplot.savefig("hplot_case.svg", format="svg")
```

### Plotting to an existing axis

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
hplot.plot(
    ci_show=True,
    ax=ax,
    display_base_type="tumor",        # shown in axis labels and title
    display_target_type="lymphocytes",
)
plt.tight_layout()
plt.savefig("hplot.png", dpi=300)
```

### `HPlot.fit()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Input data frame. |
| `target_prop_col` | `str` | — | Column for the target cell proportion. |
| `layer_col` | `str` | — | Column for the layer index. |
| `group_col` | `str \| None` | `None` | Column to split into separate lines. |
| `base_prop_col` | `str \| None` | `None` | Column for the base cell proportion (overlaid line). |
| `distance_col` | `str \| None` | `None` | Column for mean physical distance per layer. |
| `distance_unit` | `str \| None` | `None` | Unit label shown on the x-axis (e.g. `"µm"`). |
| `ci` | `float` | `0.95` | Confidence level. Uses t-distribution for n ≤ 30, z-distribution for n > 30. |
| `color_map` | `dict \| None` | `None` | Explicit `{label: color}` mapping. Overrides `palette`. |
| `palette` | `sequence \| None` | `None` | Color sequence. Defaults to `plt.cm.tab10`. |
| `legend_order` | `list \| None` | `None` | Order of legend entries. |
| `legend_title` | `str \| None` | `None` | Title for the legend box. |
| `legend_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to `ax.legend()`. |

### `HPlot.plot()` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ci_show` | `bool` | `True` | Whether to draw shaded confidence interval bands. |
| `ax` | `Axes \| None` | `None` | Existing matplotlib axis to draw into. |
| `display_base_type` | `str` | `"tumor"` | Name of the reference tissue (used in title and x-axis label). |
| `display_target_type` | `str` | `"immune cells"` | Name of the target cell type (used in y-axis label). |

---

## Batch CLI

```bash
python run_hplot.py \
  --input input.csv \
  --target_prop_col target_prop \
  --base_prop_col base_prop \
  --layer_col layer \
  --group_col subtype \
  --distance_col distance \
  --distance_unit µm \
  --output_dir hplots \
  --file_prefix case \
  --file_format svg \
  --dpi 300 \
  --ci
```

The CLI reads the CSV, groups by `--group_col` (if provided), and saves one H-Plot file per group into `--output_dir`.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Path to input CSV file. |
| `--target_prop_col` | `target_prop` | Column for target cell proportion. |
| `--base_prop_col` | `None` | Column for base cell proportion (optional). |
| `--layer_col` | `layer` | Column for layer index. |
| `--group_col` | `None` | Column to split into separate output files. |
| `--distance_col` | `None` | Column for physical distance per layer. |
| `--distance_unit` | `None` | Physical distance unit label (e.g. `µm`). |
| `--output_dir` | `hplots` | Directory for output files. |
| `--file_prefix` | `hplot` | Prefix for output filenames. |
| `--file_format` | `svg` | Output format: `svg`, `pdf`, or `png`. |
| `--dpi` | `300` | DPI for raster output (PNG). |
| `--ci` | flag | Show confidence interval bands. |

---

## Batch Python API

```python
from hplot.runners import run_hplot_batch

run_hplot_batch(
    df=df,
    target_prop_col="target_prop",
    base_prop_col="base_prop",       # optional
    layer_col="layer",
    group_col="subtype",
    distance_col="distance",
    distance_unit="µm",
    ci=0.95,
    output_dir="hplots",
    file_prefix="case",
    ci_show=True,
    file_format="svg",
    dpi=300,
)
```

---

## How confidence intervals are computed

For each layer, `hplot` aggregates the proportion values across all cases in a group and computes:

- **Mean** proportion across cases
- **Standard error of the mean** (SEM)
- **CI bounds** using:
  - t-distribution (two-tailed) when n ≤ 30
  - z-distribution when n > 30

When a layer contains only a single case (n = 1), CI bounds equal the mean (no interval shown).

---

## Project structure

```
hplot/
  core.py      — HPlot class (fit / plot / savefig)
  plotting.py  — plot_hplot() rendering function
  stats.py     — compute_layer_stats() per-layer CI computation
  runners.py   — run_hplot_batch() batch helper
  cli.py       — argparse CLI entry point
run_hplot.py   — convenience script entry point
```

---

## License

MIT License
